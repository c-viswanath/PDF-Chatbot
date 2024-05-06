import os
import streamlit as st
from langchain import hub
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# Load ChatGPT model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Function to read documents
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define user and AI icons
user_icon = "👤"
ai_icon = "🤖"

# Sidebar layout for file upload
st.sidebar.title("Upload PDF")
uploaded_files = st.sidebar.file_uploader("Choose PDF file/files", accept_multiple_files=True)
for uploaded_file in uploaded_files or []:
    if uploaded_file is not None:
        file_name = uploaded_file.name
        save_directory = "documents/"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        save_path = os.path.join(save_directory, file_name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"File '{file_name}' uploaded successfully!")

# Main content layout
st.title("🤖PDF Chatbot🤖")

# Display chat history
all_messages = st.session_state.messages # Display both user and AI messages
for message in all_messages:
    with st.container():
        with st.chat_message(message["role"]):
            st.write(message["prompt"])
        with st.chat_message("assistant"):
            st.write(message["response"])
        # if message['role'] == 'user':
        #     st.markdown(
        #         f"<div style='display: flex; align-items: center;'>"
        #         f"<div>{user_icon}</div>"
        #         f"<div>{message['prompt']}</div>"
        #         f"</div>",
        #         unsafe_allow_html=True
        #     )
        # if message['response']:
        #     st.markdown(
        #         f"<div style='display: flex; align-items: flex-start;'>"
        #         f"<div>{ai_icon}</div>"
        #         f"<div>{message['response']}</div>"
        #         f"</div>",
        #         unsafe_allow_html=True
        #     )
# st.write(st.session_state.messages)
# Main functionality
status = "Failed"
if uploaded_files:
    status = "Success"
    docs = read_doc('documents/')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    prompt = st.text_input("Ask your question")
    if prompt:
        response = rag_chain.invoke(prompt)
        st.session_state.messages.append({"role": "user", "prompt": prompt, "response": response})

        with st.container():
                st.markdown(
                    f"<div style='display: flex; align-items: flex-start;'>"
                    f"<div>{ai_icon}</div>"
                    f"<div>{response}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        rag_chain.invoke(" ")
# CSS for pinning chat bar to bottom of the screen
st.markdown(
    """
    <style>
    .stInput {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        z-index: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)
