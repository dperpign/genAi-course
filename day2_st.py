import streamlit as st
import os
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace



llm = HuggingFaceEndpoint( \
# repo_id="huggingfaceh4/zephyr-7b-alpha",  # 7B model on mistral
    repo_id="microsoft/Phi-3-mini-4k-instruct",  #3.8B model
    huggingfacehub_api_token=os.environ['OPENAI_KEY']
)

chat = ChatHuggingFace(llm=llm, verbose=True)
st.title('my first gen AI app2222')

st.markdown("""
this is a sample markdown
try it on your 'own'
""")

openai_api_key = st.sidebar.text_input("AI key")
name = st.text_input("enter some text", "enter here")
option = st.radio("choose one option:", options = ["Option1", "Option2"], index=0)

value = st.slider("Enter a value: ", 0, 100, 20)
print(value)
print(option)

def gen_response(txt):
    # llm = OpenAI(temperature = 0.7, openai_api_key=openai_api_key)
    # st.info(llm.inkove(txt))
    ans = chat.invoke(txt)
    st.info(ans.content)

with st.form("sample app"):
    txt = st.text_area("enter text:", "what GPT stands for")
    subm = st.form_submit_button("submit")
    if subm:
        gen_response(txt)


