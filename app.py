import os
import google.generativeai as genai
import streamlit as st
import tempfile

# Import all the necessary LangChain components, using the modern, non-deprecated paths
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.document_loaders import GoogleDriveLoader # Correct import
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

# --- App Configuration ---
st.set_page_config(page_title="Chat with your Data", page_icon="📚")
st.title("📚 Chat with Websites, PDFs, or Google Drive")
st.write("Select a data source, provide your input, and ask questions about the content.")

# --- Google API Configuration ---
try:
    # Using Streamlit's secrets management
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except (KeyError, TypeError):
    st.error("Could not find Google API Key in Streamlit secrets. Please add it to .streamlit/secrets.toml")
    st.stop()

# --- Caching the RAG Pipeline ---
# This function loads data, processes it, and sets up the QA chain.
# The @st.cache_resource decorator ensures this heavy lifting is done only once per source.
@st.cache_resource
def load_and_process_data(source_type, source_input):
    """
    Loads data based on source type (URL, PDF, or Google Drive),
    processes it, and returns the QA chain and retriever.
    The source_input is used as the key for caching.
    """
    if source_type == "URL":
        st.write(f"Learning from: {source_input}")
        loader = WebBaseLoader(source_input)
        pages = loader.load_and_split()
    
    elif source_type == "PDF":
        # For uploaded files, save to a temporary file to get a path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(source_input.read())
            tmp_file_path = tmp_file.name
        st.write(f"Learning from PDF: {source_input.name}")
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()
        os.remove(tmp_file_path) # Clean up the temporary file

    elif source_type == "Google Drive":
        st.write(f"Learning from Google Drive Folder: {source_input}")
        # Use the robust loader configuration with explicit paths
        loader = GoogleDriveLoader(
            folder_id=source_input,
            credentials_path="credentials.json",
            token_path="token.json",
            recursive=False
        )
        pages = loader.load()

    # Common processing steps for all sources
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    texts = text_splitter.split_documents(pages)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(texts, embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    chain = load_qa_chain(llm, chain_type="stuff")
    
    return chain, vector_store.as_retriever()

# --- User Interface and Interaction ---

# Let user choose the data source
source_choice = st.radio(
    "Choose your data source:",
    ("Website URL", "Upload PDF", "Google Drive Folder"),
    horizontal=True
)

# Initialize session state to hold the chain, retriever, and current source key
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
    st.session_state.retriever = None
    st.session_state.source_key = None

# --- Logic for each data source ---
if source_choice == "Website URL":
    url_input = st.text_input("Enter the website URL:", placeholder="https://example.com")
    if url_input and st.session_state.source_key != url_input:
        with st.spinner("Learning from the website... This may take a moment."):
            st.session_state.qa_chain, st.session_state.retriever = load_and_process_data("URL", url_input)
            st.session_state.source_key = url_input
        st.success("Website learned! You can now ask questions.")

elif source_choice == "Upload PDF":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file and st.session_state.source_key != uploaded_file.name:
        with st.spinner("Learning from the PDF... This may take a moment."):
            st.session_state.qa_chain, st.session_state.retriever = load_and_process_data("PDF", uploaded_file)
            st.session_state.source_key = uploaded_file.name
        st.success("PDF learned! You can now ask questions.")

elif source_choice == "Google Drive Folder":
    folder_id_input = st.text_input("Enter your Google Drive Folder ID:")
    if folder_id_input and st.session_state.source_key != folder_id_input:
        st.info("You may need to authenticate with Google in a new browser tab the first time this runs.")
        with st.spinner("Learning from Google Drive folder... This can take a while for many documents."):
            st.session_state.qa_chain, st.session_state.retriever = load_and_process_data("Google Drive", folder_id_input)
            st.session_state.source_key = folder_id_input
        st.success("Google Drive folder learned! You can now ask questions.")

# --- Question Answering Logic (runs if a source has been loaded) ---
if st.session_state.qa_chain and st.session_state.retriever:
    st.divider()
    user_question = st.text_input("Ask a question about the loaded content:", placeholder="e.g., What is the main topic?")

    if user_question:
        with st.spinner("Thinking..."):
            relevant_docs = st.session_state.retriever.get_relevant_documents(user_question)
            response = st.session_state.qa_chain.invoke({"input_documents": relevant_docs, "question": user_question})
            st.subheader("Answer:")
            st.write(response["output_text"])
