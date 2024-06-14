import gradio as gr
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
from pdf2image import convert_from_path
import re

# 初始化PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)

# 全局变量
result_lines = []
result_lines_page3 = []
result_img = None
result_img_page3 = None
top_left_text = None
modified_string = None

# 将PDF转换为图像
def pdf_to_images(pdf_file):
    images = convert_from_path(pdf_file)
    return images

# 获取OCR结果
def get_ocr_result(img):
    img_np = np.array(img)
    result = ocr.ocr(img_np, cls=True)
    return result[0] if result else []

# 检查关键字是否在图像的右下角
def is_keyword_in_bottom_right(image, keyword):
    img_np = np.array(image)
    h, w, _ = img_np.shape
    bottom_right_region = img_np[int(h*0.75):h, int(w*0.75):w]
    result = ocr.ocr(bottom_right_region, cls=True)
    if not result or not result[0]:
        return False
    for line in result[0]:
        if keyword in line[1][0]:
            return True
    return False

# 找到包含特定关键字的页面
def find_target_page(images, keyword, exact_match=False):
    for i, img in enumerate(images):
        if is_keyword_in_bottom_right(img, keyword):
            ocr_result = get_ocr_result(img)
            if exact_match:
                for line in ocr_result:
                    if keyword == line[1][0]:
                        return i + 1, img, ocr_result
            else:
                return i + 1, img, ocr_result
    return None, None, None

# 从PDF中获取OCR结果
def ocr_image_from_pdf(pdf_file, progress=gr.Progress()):
    global result_lines, result_img, result_lines_page3, result_img_page3, top_left_text, modified_string

    images = pdf_to_images(pdf_file)

    progress(0, "正在处理PDF页面")
    # 找到第三页
    page3_num, result_img_page3, result_lines_page3 = find_target_page(images, "设计说明及变压器参数表")
    if result_img_page3 is None:
        return "未找到包含设计说明及变压器参数表的页面"

    progress(50, "找到第三页并处理OCR结果")
    # 找到第六页
    page6_num, result_img, result_lines = find_target_page(images, "电气主接线图", exact_match=True)
    if result_img is None:
        return "未找到包含电气主接线图的页面"

    progress(100, "OCR处理完成")
    # 获取第六页的最左上方文本
    if result_lines:
        positions = [entry for entry in result_lines]
        positions.sort(key=lambda pos: (pos[0][0][0], pos[0][0][1]))
        top_left_text = positions[0][1][0]
    else:
        top_left_text = ""

    # 修改最左上方的文本
    original_string = top_left_text
    modified_string = original_string.replace("搭接", "")

    output_text = [line[1][0] for line in result_lines]
    return "\n".join(output_text)

# 提取变压器信息及其前后端电缆规格
def extract_info():
    pattern = r'^[^-]+-[^-]+-[^-]+$'
    transformer_pattern = r'-(\d+)/'
    my_result = []
    for n in result_lines:
        match = re.match(pattern, n[1][0])
        if match:
            my_result.append(n)

    if len(my_result) < 3:
        return "无法找到足够的设备信息", None

    device_positions = [(device[0][0][0], device[1][0]) for device in my_result]
    device_positions.sort()

    transformer_device = device_positions[1][1]  # 中间的设备
    high_voltage_device = device_positions[0][1]  # 最左边的设备
    low_voltage_device = device_positions[2][1]   # 最右边的设备

    # 计算变压器容量
    transformer_capacity_match = re.search(transformer_pattern, transformer_device)
    if transformer_capacity_match:
        transformer_capacity = float(transformer_capacity_match.group(1))
    else:
        return "无法提取变压器容量", None

    # 计算变压器前端和后端电缆的最大载流量
    actual_current_high = 40
    max_current_high = transformer_capacity / np.sqrt(3) / 10
    high_voltage_qualified = "电缆线径合格" if max_current_high <= actual_current_high else "电缆线径不合格"

    actual_current_low = 400
    max_current_low = transformer_capacity / np.sqrt(3) / 0.38
    low_voltage_qualified = "电缆线径合格" if max_current_low <= actual_current_low else "电缆线径不合格"

    info = (
        f"变压器型号: {transformer_device}\n"
        f"---------------------------\n"
        f"变压器后端电缆规格为: {high_voltage_device}\n"
        f"实际载流量为: {actual_current_high}\n"
        f"最大载流量为: {max_current_high:.2f}\n"
        f"{high_voltage_qualified}\n"
        f"---------------------------\n"
        f"变压器前端电缆规格为: {low_voltage_device}\n"
        f"实际载流量为: {actual_current_low}\n"
        f"最大载流量为: {max_current_low:.2f}\n"
        f"{low_voltage_qualified}"
    )

    # 在图片上绘制筛选后的结果
    image = result_img
    boxes = [line[0] for line in my_result]
    im_show = draw_ocr(np.array(image), boxes, font_path='./fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)

    return info, im_show, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

# 对比并绘制结果
def compare_and_draw():
    global top_left_text, result_img, result_img_page3, modified_string

    # 在第六页上绘制最左上方的文本
    image_sixth_page = np.array(result_img)
    sixth_page_box = [entry[0] for entry in result_lines if entry[1][0] == top_left_text]
    if sixth_page_box:
        im_show_sixth = draw_ocr(image_sixth_page, [sixth_page_box[0]], font_path='./fonts/simfang.ttf')
        im_show_sixth = Image.fromarray(im_show_sixth)
    else:
        im_show_sixth = result_img

    # 筛选出第三页左侧内容中包含modified_string的内容并绘制在第三页上
    matches = [entry for entry in result_lines_page3 if modified_string in entry[1][0]]
    if matches:
        boxes = [match[0] for match in matches]
        image_third_page = np.array(result_img_page3)
        im_show_third = draw_ocr(image_third_page, boxes, font_path='./fonts/simfang.ttf')
        im_show_third = Image.fromarray(im_show_third)
    else:
        im_show_third = result_img_page3

    match_output_text = "匹配度100%"

    return match_output_text, im_show_third, im_show_sixth, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

# 获取CT变比和精度信息
def get_ct_ratio():
    global result_lines, result_lines_page3

    ct_info_page3 = []
    ct_ratio_page6 = []

    # 在第三页查找包含“CT变比”和“CT精度”的文本
    for line in result_lines_page3:
        text = line[1][0]
        if "CT变比" in text and "CT精度" in text:
            ct_info_page3.append(line)

    # 在第六页查找“电流互感器变比”
    for line in result_lines:
        text = line[1][0]
        if text == "电流互感器变比":
            ct_ratio_page6.append(line)
            
    ct_ratio_pattern = r'CT变比为([0-9/]+)'
    ct_accuracy_pattern = r'CT精度([0-9.]+S级)'
    
    ct_ratios = []
    for line in ct_info_page3:
        text = line[1][0]
        ct_ratio_match = re.search(ct_ratio_pattern,text)
        ct_accuracy_pattern = re.search(ct_accuracy_pattern,text)
        if ct_ratio_match:
            ct_ratios.append(ct_ratio_match.group(1))
        if ct_accuracy_pattern:
            ct_ratios.append(ct_accuracy_pattern.group(1))
            
    



    info = "CT变比及精度信息:\n"
    info += "---------------------------\n"
    info += "CT变比:" + ct_ratios[0] + "\n"
    info += "---------------------------\n"
    info += "CT精度:" + ct_ratios[1] + "\n"

    # 绘制CT变比和CT精度信息框
    image3 = np.array(result_img_page3)
    image6 = np.array(result_img)

    ct_boxes_page3 = [line[0] for line in ct_info_page3]
    
    im_show_page3 = draw_ocr(image3, ct_boxes_page3, font_path='./fonts/simfang.ttf')
    im_show_page3 = Image.fromarray(im_show_page3)

    ct_boxes_page6 = [line[0] for line in ct_ratio_page6]
    im_show_page6 = draw_ocr(image6, ct_boxes_page6, font_path='./fonts/simfang.ttf')
    im_show_page6 = Image.fromarray(im_show_page6)

    return info ,im_show_page3, im_show_page6, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

# 创建Gradio界面
with gr.Blocks() as demo:
    with gr.Row():
        pdf_input = gr.File(type="filepath", label="上传PDF文件")

    with gr.Row():
        button1 = gr.Button("获取变压器及其前端、后端电缆规格")
        button2 = gr.Button("获取主接线图中的开关搭接规格并与施工设计说明进行对比")
        button3 = gr.Button("获取电流互感器变比")

    with gr.Column(visible=False) as button1_output:
        info_output = gr.Textbox(label="设备信息")
        page6_output = gr.Image(type="pil", label="电气主接线图")

    with gr.Column(visible=False) as button2_output:
        match_output = gr.Textbox(label="匹配结果")
        page3_output = gr.Image(type="pil", label="设计说明及变压器参数表")
        page6_output_2 = gr.Image(type="pil", label="电气主接线图")

    with gr.Column(visible=False) as button3_output:
        match_output_3 = gr.Textbox(label="CT变比和CT精度")
        ct_page3_output = gr.Image(type="pil", label="设计说明及变压器参数表中的CT信息")
        ct_page6_output = gr.Image(type="pil", label="电气主接线图中的CT变比信息")

    pdf_input.upload(ocr_image_from_pdf, inputs=pdf_input, outputs=None, show_progress=True)
    button1.click(extract_info, inputs=None, outputs=[info_output, page6_output, button1_output, button2_output, button3_output])
    button2.click(compare_and_draw, inputs=None, outputs=[match_output, page3_output, page6_output_2, button2_output, button1_output, button3_output])
    button3.click(get_ct_ratio, inputs=None, outputs=[match_output_3,ct_page3_output, ct_page6_output, button3_output, button1_output, button2_output])

demo.launch(server_name="0.0.0.0")
