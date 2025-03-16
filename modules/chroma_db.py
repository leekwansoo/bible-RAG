# import dependancies
# url: https://medium.com/ai-simplified-in-plain-english/making-a-rag-app-with-deepseek-r1-ollama-4a7d634b97a0
## Making a RAG App with DeepSeek-R1 & Ollama

import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate

EMBEDDING_MODEL = "qwen2"

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_db = InMemoryVectorStore(embeddings)

llm = OllamaLLM(model=EMBEDDING_MODEL)

# Handy Helper Functions

def add_to_vectostore(chunks):
    vector_db.add_documents(chunks)
    return ("Data stored in Vectorstore")
    
def find_related_docs(query):
    return vector_db.similarity_search(query)

def generate_answer(user_query, related_docs):
    context = "\n\n".join([d.page_content for d in related_docs])

    prompt_template = """
        You are a research assistant. Use the context below to answer the user's questionin korean.
        Context: {context}
        Question: {question}
        Answer:
    """
    chat_prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = chat_prompt | llm

    final_answer = chain.invoke({
        "context": context,
        "question": user_query
    })

    return final_answer
