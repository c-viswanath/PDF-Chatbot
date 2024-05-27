# PDF Chatbot with Streamlit and LangChain

This project creates a PDF chatbot using Streamlit and LangChain. The chatbot can read and understand uploaded PDF files, allowing users to ask questions and receive answers based on the content of those documents.

## Prerequisites

Ensure you have the following libraries installed:
- `os`
- `streamlit`
- `langchain`
- `langchain_chroma`
- `langchain_core`
- `langchain_openai`

You can install the required libraries using pip:
```bash
pip install streamlit langchain langchain_chroma langchain_core langchain_openai
```
## Project Structure

- **Main functionality**:
    - Upload PDF files.
    - Load and read the content of PDF files.
    - Process the content and create embeddings.
    - Ask questions based on the content and get responses from the chatbot.

## Code Explanation

### 1. Load ChatGPT Model

The `ChatOpenAI` model is loaded to generate responses based on user queries.
```python
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
```

### 2. Read Documents

A function to read documents from a specified directory.
```python
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents
```

### 3. Format Documents

A function to format the documents into a single string.
```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```

### 4. Initialize Session State

Initialize the session state to keep track of messages.
```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

### 5. Sidebar Layout for File Upload

Create a sidebar for file uploads.
```python
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
```

### 6. Main Content Layout

Display the chat history and handle user inputs.
```python
st.title("🤖PDF Chatbot🤖")

all_messages = st.session_state.messages
for message in all_messages:
    with st.container():
        with st.chat_message(message["role"]):
            st.write(message["prompt"])
        with st.chat_message("assistant"):
            st.write(message["response"])
```

### 7. Main Functionality

Load documents, create embeddings, and handle user queries.
```python
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
```

### 8. CSS for Chat Bar

CSS to pin the chat bar to the bottom of the screen.
```python
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
```

## How to Run

1. Ensure all prerequisites are installed.
2. Save the script as `pdf_chatbot.py`.
3. Run the script using Streamlit:
    ```bash
    streamlit run pdf_chatbot.py
    ```
4. Upload your PDF files through the sidebar.
5. Ask questions using the text input at the bottom of the page.

## Conclusion

This project demonstrates how to create a chatbot that can understand and respond to questions based on the content of uploaded PDF documents. By leveraging Streamlit and LangChain, we can build an interactive and user-friendly application.
```
