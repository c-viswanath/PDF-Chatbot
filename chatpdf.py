import dotenv
dotenv.load_dotenv()
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

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


## Lets Read the document
def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#create storage
if 'messages' not in st.session_state:
    st.session_state.messages = []

user_icon = "👤"
ai_icon = "🤖"

#display chat history
st.write(st.session_state.messages)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["prompt"])
    with st.chat_message("assistant"):
        st.write(message["response"])


##streamlit app
def main():
    status="Failed"
    try:
        uploaded_files = st.file_uploader("Choose PDF file/files", accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                # Get the file name
                file_name = uploaded_file.name
                save_directory = "documents/"
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)
                save_path = os.path.join(save_directory, file_name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"File '{file_name}' uploaded successfully!")
                status="Success"      

        if status=="Success":
            docs=read_doc('documents/')            
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
            #take input from chat

            prompt=st.chat_input("Ask your question")
            if prompt:
                response=rag_chain.invoke(prompt)

                #add to storage session
                st.session_state.messages.append({"role":"user","prompt":prompt,"response":response})
                
                #display what was typed
                with st.chat_message("user"):
                    st.write(prompt)

                #display response as AI Chat
                with st.chat_message("assistant"):
                    st.markdown(response)

            #store response
            # st.session_state.messages.append({"role":"assistant","content":response})    
            # st.write(response)
            #sample qns
            #what is tobacco?
            #how does tobacco affect the body?
            #what are 1 bit llms? and how much faster as compared to conventional llms? 
    except:
        pass            


if __name__ == "__main__":
    main()