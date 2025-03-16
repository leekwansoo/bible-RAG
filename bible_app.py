import streamlit as st 
from langchain_community.document_loaders import TextLoader
import os
from modules.chroma_db import add_to_vectostore, find_related_docs, generate_answer
from file_handler import save_uploaded_file, load_pdf, split_into_chunks, generate_question, create_query_file, add_qa_file, check_file_exist

model = "qwen2" # download model with "ollama pull qwen2" from PS terminal

# Main Page content 
def main_query():
    user_query = st.text_input("Enter your question from uploaded documents:") 
    if user_query: 
        with st.spinner("Searching and generating answer..."):
            # 5. Find the relevant chunks
            related_docs = find_related_docs(user_query)
            # 6. Generate the final answer
            response = generate_answer(user_query, related_docs)
            if response:
                st.write(response)
                qa_pair = {"query": user_query, "answer": response}
                qa_file = add_qa_file(file_name, qa_pair)
                st.write(f"QA pair is saved in {qa_file}")
     
st.title("Bible Study Reader")
st.write(f"Ask questions about Bible contents from uploaded Books")

st.session_state["query_message"] = []
st.session_state["query_file"] = []

# Create a sidebar for navigation
st.sidebar.title("Menu")
options = st.sidebar.radio("Select an option", ["Upload File", "Query from Uploaded File"])

if options == "Upload File":
    st.sidebar.header("Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf", key="pdf_uploader")

    if uploaded_file is not None:
        file_name = uploaded_file.name
        dir = "data"
        check_exist = check_file_exist(dir, file_name)
        check_exist = "noexist"
        if check_exist == "noexist":
            # 1. Save the file
            path_to_pdf = save_uploaded_file(uploaded_file)
            
            # 2. Load the PDF content
            docs = load_pdf(path_to_pdf)

            # 3. Split into chunks
            chunks = split_into_chunks(docs)

            # 4. Index chunks in the vector store
            response = add_to_vectostore(chunks)
            if response:
                st.sidebar.success("PDF indexed successfully!")

                query_list = generate_question(docs)
                query_file = create_query_file(file_name, query_list)
                st.session_state["query_file"].append(query_file)
                for query in query_list:
                            st.session_state["query_message"].append(query)
                            st.sidebar.markdown(query)
            else: 
                    st.sidebar.write("storing PDF file into vector store failed")               
        else:
            st.sidebar.write("Please upload a PDF and select documents to get started.")                
            
            
else:
    options = "Query from Uploaded File"
    st.header("Query from Uploaded File")
    
    query_file_list = os.listdir("query")
    selected = st.sidebar.selectbox("Select document to query", query_file_list)
    file_name = f"query/{selected}"
    
    loader =TextLoader(file_name, encoding = "utf-8")
    documents = loader.load()
    query_list = documents[0].page_content.split("\n")
    # Print the list
    i = 0
    for query in query_list:
        i += 1
        if query is not None:
            st.sidebar.markdown(query)
    
            if st.sidebar.button(f"Query", key=f"button_{i}"):  # Add a button with a unique key
                related_docs = find_related_docs(query)
                response = generate_answer(query, related_docs)
                if response:
                    st.write(response)
                    qa_pair = {"query": query, "answer": response}
                    qa_file = add_qa_file(file_name, qa_pair)
                    st.write(f"QA pair is saved in {qa_file}")
    
main_query()    