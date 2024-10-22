

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
在BlogScribe01目录下
在运行app.py之前，请确保您先运行了
requirements.txt


```bash
 git clone https://github.com/PoolBee/BlogScribe01.git
```

```bash
pip install -r requirements.txt

```

# 界面&使用
<img width="473" alt="图片1" src="https://github.com/user-attachments/assets/ee178fb8-8903-4911-9ba3-619ebda8d5bd">

## 上传区
点击Browse files后选择文件上传，上传知识库词库支持(PDF、TXT、DOCX)格式
点击Browse files后选择文件上传，上传语音文件支持(wav、mp3、ogg)格式
## 处理区
- 1.点击`语音识别`，进行语音识别，界面会显示`语音转文本结果`。
- 2.点击`相似文本检索`，BlogScribe会进行相似文本检索,在**语音识别文本**与**知识库词库**中进行相似词语检索，并替换语音识别文本中识别错误词语，替换后输出在`替换后的文本`中。
- 3.点击润色文本，会对处理后的文本进行添加标点符号、标题提取、语言润色，并输出到`润色后的文本`文本框中。
- 4.点击`生成Markdown格式的博客`对润色后的文本进行Markdown格式的输出。
## 下载区
点击`下载Markdown文件`进行.md文档格式下载。


```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 6000

```
