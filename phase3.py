import os
import google.generativeai as genai

# THE FIX: Import from the new, correct package as the warning suggests.
from langchain_google_community.document_loaders import GoogleDriveLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

# --- 1. Configuration ---
# IMPORTANT: Replace these placeholders with your actual information
os.environ["GOOGLE_API_KEY"] = "AIzaSyAxNFgk2kmVcXnI5iYPhEQ_4L1uiVvZjdw"
FOLDER_ID = "1r_AJ8vVsvSpqes8uFzPnMGggabP60daY"

# Configure the Gemini client
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except (KeyError, TypeError):
    print("ERROR: GOOGLE_API_KEY not found. Please set it in the script.")
    exit()

# --- 2. Load Documents from Google Drive ---
# This step requires the 'credentials.json' and 'token.json' files.
print(f"Loading documents from Google Drive folder: {FOLDER_ID}")
try:
    # This now uses the modern, non-deprecated loader
    loader = GoogleDriveLoader(
        folder_id=FOLDER_ID,
        credentials_path="credentials.json",
        token_path="token.json",
        recursive=False
    )
    pages = loader.load()
    print(f"Successfully loaded {len(pages)} pages from all documents in the folder.")
except Exception as e:
    print(f"ERROR: Could not load from Google Drive. Please check your Folder ID and ensure 'credentials.json' is in the same folder as this script.")
    print(f"Details: {e}")
    exit()

# --- 3. Split Documents into Chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
texts = text_splitter.split_documents(pages)
print(f"Split content into {len(texts)} chunks.")

# --- 4. Create Embeddings and Store in ChromaDB ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("Creating vector store... (This might take a moment)")
vector_store = Chroma.from_documents(texts, embeddings).as_retriever()
print("Vector store created.")

# --- 5. Setup the LLM and QA Chain ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
chain = load_qa_chain(llm, chain_type="stuff", verbose=False)
print("QA Chain ready.")

# --- 6. Ask a Question and Get an Answer ---
# IMPORTANT: Change this question to be relevant to your documents.
query = "How many total credits are required at the end of 4 years?"
print(f"\nAsking question: '{query}'")

relevant_docs = vector_store.get_relevant_documents(query)
print("Generating answer...")
response = chain.invoke({"input_documents": relevant_docs, "question": query})

# --- 7. Print the Final Answer ---
print("\n--- Answer ---")
print(response["output_text"])
print("---------------")
