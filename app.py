import streamlit as st
import pandas as pd
import os
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(layout="wide", page_icon="💬", page_title="HKSTP🤖")

st.markdown(
            f"""
            <h1 style='text-align: center;'> Search for the related companies in HKSTP 😁</h1>
            """,
            unsafe_allow_html=True,
        )
# Sidebar inputs
apikey = st.sidebar.text_input("Please input the API_KEY:", type='password')
df = pd.read_csv("HKSTP_Company.csv")
Key_words = st.sidebar.text_input("Please input the key words:")
Top_key = st.sidebar.number_input("Please input the top number:", min_value=1)

# Columns to select for display
columns = ['cluster', 'cluster_TC', 'cluster_SC', 'telephone', 'fax', 'email',
       'name_EN', 'website', 'INFO', 'SUMMAR', 'PRIMARY', 'SUB', 'address_EN',
       'contact_EN', 'introduction_EN', 'product_EN', 'name_TC', 'address_TC',
       'contact_TC', 'introduction_TC', 'product_TC', 'name_SC', 'address_SC',
       'contact_SC', 'introduction_SC', 'product_SC']
all_columns = ['name_EN','introduction_EN', 'product_EN','website',]  # Add all columns you might want to display
columns_to_display = st.sidebar.multiselect("Select columns to display:", columns, default=all_columns)

if apikey:
    # Set the environment variable
    os.environ['OPENAI_API_KEY'] = apikey
    # Preload data and settings
    db = FAISS.load_local("faiss_index",OpenAIEmbeddings())
    # Initialize retriever
    retriever = db.as_retriever(search_kwargs={"k": Top_key})      
if Key_words:       
    row = []
    docs = retriever.get_relevant_documents(Key_words)
    
    if len(docs) == 0:
        st.write("No relevant documents found.")
    else:
        for i in range(min(Top_key, len(docs))):
            num = docs[i].metadata['row']
            row.append(num)
        
        result = df.loc[row][columns_to_display]
        
        st.write("Here is the result DataFrame:")
        st.write(result)
