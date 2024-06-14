# README

## 项目简介

本项目基于 `Gradio` 和 `PaddleOCR` 实现了一个 PDF 解析与 OCR 识别的应用程序。该程序能够将 PDF 文件中的指定页面转换为图像，并对图像进行 OCR 识别，提取文本信息。主要功能包括：
1. 提取变压器及其前端、后端电缆规格。
2. 获取主接线图中的开关搭接规格并与施工设计说明进行对比。
3. 获取电流互感器变比及精度信息。

##  使用方法

### 	1. 启动应用程序

运行以下命令启动 Gradio 应用：

```bash
python app.py
```

### 	2. 上传 PDF 文件

在 Gradio 界面中上传需要解析的 PDF 文件。

### 	3. 执行功能

界面提供了三个按钮，分别对应三个主要功能：
- 获取变压器及其前端、后端电缆规格
- 获取主接线图中的开关搭接规格并与施工设计说明进行对比
- 获取电流互感器变比及精度信息

点击相应按钮即可执行相应功能，并在界面中显示结果。


## 设计思路
TODO
## End

本项目实现了一个基于 OCR 技术的 PDF 解析工具，通过 Gradio 提供简洁友好的用户界面，方便用户进行交互和使用。希望本项目对您的工作有所帮助。
