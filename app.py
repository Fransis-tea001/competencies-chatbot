import os

import streamlit as st
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# initialize chat history session
if "messages" not in st.session_state:
    st.session_state.messages = []

def load_dependencies():
    from langchain_community.document_loaders.csv_loader import CSVLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    return CSVLoader, HuggingFaceEmbeddings, Chroma

@st.cache_resource(show_spinner=False)
def initalize_retriever(dataset_path: str):
    # load depencies
    CSVLoader, HuggingFaceEmbeddings, Chroma = load_dependencies()
    
    # load dataset (csv file) as documents 
    loader = CSVLoader(file_path=dataset_path)
    docs = loader.load()

    # initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # create chroma vector store
    vector_store = Chroma(
    embedding_function=embeddings,
    )

    # add document in vector store
    vector_store.add_documents(documents=docs)
    
    # initialize vector store as retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    return retriever

@st.cache_resource(show_spinner=False)
def connect_llm(model_name: str):
    llm = ChatGroq(model_name=model_name, temperature=0.5, api_key=GROQ_API_KEY)
    return llm

@st.cache_resource(show_spinner=False)
def retrieval_chain_initialize(_retriever, _llm):
    retrieval_chain = (
        {
            "context": _retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | ChatPromptTemplate.from_template(
            """You're a helpful and encouraging AI coach. Based on the user's question and the relevant course data below, do the following:
            
            1. Choose the most relevant competency the user should focus on.
            2. Recommend a course that fits this competency, and explain it clearly.
            
            Format your answer like this:
        
            ### [Competency Name]  
            Write a friendly and encouraging sentence that explains why this competency is suitable for the user in natural tone.

            **Course Recommendation**  
            **Competency:** [Competency Name]  
            **Course Name:** [Course Title]  
            **Description:**  
            A short, clear explanation of the course content. Write like you're helping the user understand why this course is valuable.

            ---
        
            Here is the information you'll use:
            Search Result:
            {context}

            User Question:
            {question}

            Answer:"""
        )
        | _llm
    ).with_types(output_type=str)
    return retrieval_chain

def format_docs(docs):
  formatted_docs = "\n\n".join(doc.page_content for doc in docs)
  return formatted_docs

def main():
    # create and populate vector store
    try:
        dataset_path = os.path.join("data", "openai_result_cleaned.csv")
        retriever = initalize_retriever(dataset_path=dataset_path)
    except Exception as e:
        st.error(f"Cannot create vector database: {e}")

    # connect llm model
    try:
        llm = connect_llm(model_name="llama-3.1-8b-instant")
    except Exception as e:
        st.error(f"Cannot connect with llm: {e}")
    
    # define retrieval chain
    retrieval_chain = retrieval_chain_initialize(retriever, llm)

    # display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # accept user input
    if user_prompt := st.chat_input("What skill do you want to learn?"):
        # add user message to chat history and display message in chat container
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # retrieve relevant data and dispay it in chat container
        with st.chat_message("assistant"):
            try:
                response = retrieval_chain.invoke(user_prompt)
            except Exception as e:
                st.error(f"Cannot retreive llm response: {e}")
            
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            st.markdown(response.content)

if __name__ == "__main__":
    main()