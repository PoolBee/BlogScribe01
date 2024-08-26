import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import Any, List, Optional
import whisper

import tempfile
from modelscope import snapshot_download

# 向量模型下载
model_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5', cache_dir='./')

# 源大模型下载
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')

# 定义模型路径
model_path = './IEITYuan/Yuan2-2B-Mars-hf'

# 定义向量模型路径
embedding_model_path = './AI-ModelScope/bge-small-zh-v1___5'

# 定义模型数据类型
torch_dtype = torch.bfloat16 # A10

class Yuan2_LLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, mode_path: str):
        super().__init__()
        print("Creat tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
        self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>',
                                   '<commit_before>', '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>',
                                   '<jupyter_code>', '<jupyter_output>', '<empty_output>'], special_tokens=True)

        print("Creat model...")
        self.model = AutoModelForCausalLM.from_pretrained(mode_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        prompt = prompt.strip()
        prompt += "<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        outputs = self.model.generate(inputs, do_sample=False, max_length=4096)
        output = self.tokenizer.decode(outputs[0])
        response = output.split("<sep>")[-1].split("<eod>")[0]
        return response

    @property
    def _llm_type(self) -> str:
        return "Yuan2_LLM"

class TextCorrector:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.template = """

我希望你能充当博客文章润色助手。最终仅输出润色后的文章，请检查以下文本并对第一步句子增加标点，第二步对句子结构、语法、用词和表达清晰度等进行改进，目的是提高文章的质量和可读性，文本：{replace_text}`
补充：请不要输出其他的语句，仅输出文章。


"""
        self.prompt = PromptTemplate(
            input_variables=["replace_text"],
            template=self.template
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def polish_text(self, replace_text):
        polished_text = self.chain.run({
            "replace_text": replace_text
        })
        return polished_text

class TextGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["After_text"],
            template="""
请你作为Markdown专家：负责处理和生成Markdown格式的内容；
文本分析专家：负责识别和提取文本中的特定信息；
内容编辑专家：负责整理和编辑文本内容，使其符合Markdown格式。

    *根据`我的文本`{After_text}的内容，进行以下步骤：
    第一步，识别出的特定语句提取Markdown标题、日期、类型、标签，替换Markdown头部的对应字段
    第二步，对`Markdown头部`（---与---的部分），对`Markdown主体`（头部以外的部分）进行Readme文档格式修改，最终仅输出完整Markdown格式源代码。
*补充：请不要输出其他的语句，仅输出完整Markdown格式源代码。
*以下仅是一个示例：
***
`我的文本`：
大家好,今天的博客讲的是Docker的背景知识，博客标题为第一篇Docker的背景知识，博客日期2024年7月20日，博客类型为Docker，博客标签为背景知识。
传统的部署方式，往往是用一堆帮助文档，安装程序。而Docker使用打包镜像发布测试，能一键运行更便捷的升级和扩缩容,使用了Docker之后，我们部署引用就和搭积木一样！非常的简单。当项目打包为一个镜像，可以扩展到——服务器A！服务器B！这样实现了更简单的系统运维,容器化之后，我们开发、测试环境都是高度一致的，这样能实现更高效的计算资源利用。本质上Docker是内核级别的虚拟化，可以在一个物理机上运行很多的容器实例！服务器的性能能被压榨到极致。
`你的输出（完整Markdown格式源代码）`：
        ---
        title: 第一篇Docker的背景知识
        date: 2024-07-20
        categories:
        - Docker
        tags:
        - 背景知识
        ---

        # Docker的背景知识
        # 引言
        大家好,今天的博客讲的是Docker的背景知识，博客标题为第一篇Docker的背景知识。
        ***
        # Docker的优点

        ##更快速的交付和部署

        ###传统的部署方式
        往往是用一堆帮助文档，安装程序。
        ###Docker的部署方式
        而Docker使用打包镜像发布测试，能一键运行更便捷的升级和扩缩容。

        ##更便捷的升级和扩缩容
    Docker使用打包镜像发布测试，能一键运行更便捷的升级和扩缩容,使用了Docker之后，我们部署引用就和搭积木一样！非常的简单。
        ## 更便捷的升级和扩缩容
    当项目打包为一个镜像，可以扩展到——服务器A！服务器B！这样实现了更简单的系统运维,
        ## 更简单的系统运维
    容器化之后，我们开发、测试环境都是高度一致的，这样能实现更高效的计算资源利用。
        ##更高效的计算资源利用
    Docker是内核级别的虚拟化，可以在一个物理机上运行很多的容器实例！服务器的性能能被压榨到极致。
***


            """

        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def generate_text(self, text):
        generated_text = self.chain.run({
            "After_text": text
        })
        return generated_text

class KnowledgeBase:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=10,
            length_function=len
        )
        self.db = None

    def load_documents(self, file_path):
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("上传格式请以 PDF, TXT, 或 DOCX 文件.")
        documents = loader.load_and_split(self.text_splitter)
        return documents

    def vectorize_documents(self, documents):
        self.db = FAISS.from_documents(documents, self.embeddings)

    def replace_similar_texts(self, speech_text, top_k=1):
        docs = self.db.similarity_search(speech_text, k=top_k)
        similar_text = " ".join([doc.page_content for doc in docs])
# 替换逻辑
        replaced_text = speech_text
        for doc in docs:
            if doc.page_content in speech_text:
                replaced_text = replaced_text.replace(doc.page_content, doc.page_content)

        return replaced_text, similar_text

@st.cache_resource
def get_models():
    llm = Yuan2_LLM(model_path)
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return llm, embeddings

def recognize_speech(audio_file):
    temp_audio_path = tempfile.mktemp(suffix=".wav")
    with open(temp_audio_path, "wb") as temp_audio:
        temp_audio.write(audio_file.read())
    model = whisper.load_model("large-v2")
    result = model.transcribe(temp_audio_path)
    return result['text']

def main():
    st.title("🗒️BlogScribe")

    llm, embeddings = get_models()
    knowledge_base = KnowledgeBase(embeddings)
    text_corrector = TextCorrector(llm, embeddings)
    text_generator = TextGenerator(llm)

    if 'speech_text' not in st.session_state:
        st.session_state.speech_text = ""
    if 'similar_text' not in st.session_state:
        st.session_state.similar_text = ""
    if 'replaced_text' not in st.session_state:
        st.session_state.replaced_text = ""
    if 'polished_text' not in st.session_state:
        st.session_state.polished_text = ""
    if 'markdown_text' not in st.session_state:
        st.session_state.markdown_text = ""

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("上传区")
        uploaded_file = st.file_uploader("上传知识库的文件 (PDF, TXT, DOCX)", type=['pdf', 'txt', 'docx'])
        audio_file = st.file_uploader("上传语音输入的博客音频(wav, mp3, ogg)", type=['wav', 'mp3', 'ogg'])

        if uploaded_file:
            file_content = uploaded_file.read()
            temp_file_path = "temp." + uploaded_file.name.split('.')[-1]
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_content)
            docs = knowledge_base.load_documents(temp_file_path)
            knowledge_base.vectorize_documents(docs)
            st.success("文档加载并向量化完成")

        st.header("下载区")
        if st.session_state.markdown_text:
            if st.button("下载Markdown文件"):
                markdown_text = st.session_state.markdown_text
                markdown_file = tempfile.mktemp(suffix=".md")
                with open(markdown_file, "w") as file:
                    file.write(markdown_text)
                with open(markdown_file, "rb") as file:
                    st.download_button("点击下载Markdown文件", data=file, file_name="blog_post.md", mime="text/markdown")

    with col2:
        st.header("处理区")

        st.text_area("语音转文本结果", value=st.session_state.speech_text, height=150, key="speech_text_display")
        if st.button("语音识别"):
            if audio_file:
                st.session_state.speech_text = recognize_speech(audio_file)
                st.experimental_rerun()

        st.text_area("检索到的相似文本", value=st.session_state.similar_text, height=150, key="similar_text_display")
        st.text_area("替换后的文本", value=st.session_state.replaced_text, height=150, key="replaced_text_display")
        if st.button("相似文本检索替换"):
            if st.session_state.speech_text:
                replaced_text, similar_text = knowledge_base.replace_similar_texts(st.session_state.speech_text)
                st.session_state.similar_text = similar_text
                st.session_state.replaced_text = replaced_text
                st.experimental_rerun()
            else:
                st.error("请先进行语音识别")

        if st.button("润色文本"):
            if st.session_state.replaced_text:
                st.session_state.polished_text = text_corrector.polish_text(st.session_state.replaced_text)
                st.experimental_rerun()
            else:
                st.error("请先进行相似文本检索替换")

        st.text_area("润色后的文本", value=st.session_state.polished_text, height=150, key="polished_text_display")

        st.text_area("Markdown格式博客", value=st.session_state.markdown_text, height=150, key="markdown_text_display")
        if st.button("生成Markdown格式的博客"):
            if st.session_state.polished_text:
                st.session_state.markdown_text = text_generator.generate_text(st.session_state.polished_text)
                st.experimental_rerun()
            else:
                st.error("请先润色文本")

if __name__ == '__main__':
    main()