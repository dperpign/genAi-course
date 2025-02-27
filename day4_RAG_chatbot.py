import streamlit as st 
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PDFPlumberLoader  
from langchain_experimental.text_splitter import SemanticChunker  
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS  
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter

# huggingfaceh4/zephyr-7b-alpha
# microsoft/Phi-3-mini-4k-instruct
# pip install pdfplumber 
# pip install faiss-cpu

# REQS
# streamlit
# langchain
# openai
# langchain_community
# langchain_huggingface
# langchain_experimental
# pdfplumber
# faiss-cpu

from streamlit.logger import get_logger
logger = get_logger(__name__)

import os
if os.getenv('USER') == 'appuser':
    hf_token = st.secrets["HF_TOKEN"]
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token
else:
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = \
    os.environ['MY_HUGGINGFACEHUB_API_TOKEN']

def upload_file(pdf_docs):
    docs = []
    for pdf_doc in pdf_docs:
        with open("tmp.pdf", "wb") as f:
           f.write(pdf_doc.getvalue()) 

        logger.info("loading PDF")
        loader = PDFPlumberLoader("tmp.pdf")
        docs.extend(loader.load())
        logger.info("Done")

    logger.info("Semantic Chunker")
    text_splitter = CharacterTextSplitter(chunk_size=1000,
                                          chunk_overlap=10)
    documents  = text_splitter.split_documents(docs)
    logger.info("Done")

    emb = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(documents, emb)

    retriever = vector_store.as_retriever()


    llm = HuggingFaceEndpoint(repo_id = "huggingfaceh4/zephyr-7b-alpha")
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain



def main():
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    with st.sidebar:
        st.subheader("your PDF")
        pdf_docs = st.file_uploader("please upload your PDF's",
                                    type="pdf",
                                    accept_multiple_files=True
                                    )

        logger.info(pdf_docs)
        if st.button("Process"):
            with st.spinner("processing"):
                qa_chain = upload_file(pdf_docs)
                st.session_state.qa_chain = qa_chain

    user_input = st.text_input("ask your question")

    if user_input:
        with st.spinner("thinking..."):
            qa_chain = st.session_state.qa_chain
            response = qa_chain.invoke(user_input)
            st.write(response)

main()