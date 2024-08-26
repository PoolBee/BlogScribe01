---
# 详细文档见https://modelscope.cn/docs/%E5%88%9B%E7%A9%BA%E9%97%B4%E5%8D%A1%E7%89%87
domain: 领域：audio
# - cv
tags: BlogScribe
-
datasets: #关联数据集
  evaluation:
  - iic/ICDAR13_HCTR_Dataset
  test:
  #- iic/MTWI
  train:
  #- iic/SIBR
models: #关联模型
#- iic/ofa_ocr-recognition_general_base_zh

## 启动文件(若SDK为Gradio/Streamlit，默认为app.py, 若为Static HTML, 默认为index.html)
# deployspec:
#   entry_file: app.py
license: Apache License 2.0
---
#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/studios/poolbee/Chattests_Question_0520_test.git
```

# BlogScribe

个人博客已经成为技术开发者分享知识、表达观点和展示个人技能的重要平台。然而，每次书写博客时，特别是涉及到文本排版和格式处理时，往往会影响创作者的效率，使得他们在真正的内容创作上花费的时间减少。为了解决这个问题，BlogScribe诞生了。BlogScribe是一个个人博客书写助手，旨在通过语音输入自动生成高效、符合博客格式的内容。

## 功能概述

### 1. 语音识别与转文本
使用Whisper模型进行语音转文本，用户上传音频文件(wav, mp3, ogg)，进行在线的语音识别。

### 2. 知识库与语音识别
用户上传词库文档 (PDF, TXT, DOCX)，与语音识别文本通过向量模型进行比对，检索到相似的文本。

### 3. 相似文本检索替换
检索到相似的文本，对语音转文本相似词进行替换。

### 4. 润色文本及Markdown格式下载
利用Yuan2B大模型进行文本润色和格式处理，自动生成博客文章标题及小标题，确保生成的内容符合博客格式。

### 5. 博客下载
提供md格式的博客源代码下载。

## 安装与使用

### 安装依赖

在使用BlogScribe之前，请确保您先运行了
requirements.txt


```bash
 git clone https://www.modelscope.cn/studios/poolbee/BlogScribe.git
```

```bash
pip install -r requirements.txt

```

```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 6000

```