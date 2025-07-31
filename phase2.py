import os
import google.generativeai as genai

# Use WebBaseLoader for web scraping and other updated imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

# --- 1. Configuration ---
# Set your Google API Key
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY" # <-- IMPORTANT: Make sure your key is here
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# --- 2. Load the Document from a URL ---
# IMPORTANT: Replace this with the URL you want to scrape.
url = "https://nitte.edu.in/nmamit/department-electronics-communication.php"
print(f"Loading content from URL: {url}")

# Create a loader for the URL
loader = WebBaseLoader(url)
pages = loader.load_and_split()
print("Website content loaded successfully.")


# --- 3. Split the Document into Chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
context = "\n\n".join(str(p.page_content) for p in pages)
texts = text_splitter.split_text(context)
print("Text split into chunks.")


# --- 4. Create Embeddings and Store in ChromaDB ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("Creating vector store... (This might take a moment)")
vector_store = Chroma.from_texts(texts, embeddings).as_retriever()
print("Vector store created.")


# --- 5. Setup the LLM and QA Chain ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
chain = load_qa_chain(llm, chain_type="stuff", verbose=False)
print("QA Chain ready.")


# --- 6. Ask a Question and Get an Answer ---
# Define a question relevant to the webpage's content.
query = "who is the most educated professor in the ece department" # <-- CHANGE YOUR QUESTION HERE

# Find relevant documents based on the question
print(f"Searching for relevant documents for the query: '{query}'")
relevant_docs = vector_store.get_relevant_documents(query)

# Run the QA chain to get the answer
print("Generating answer...")
response = chain.invoke({"input_documents": relevant_docs, "question": query})


# --- 7. Print the Final Answer ---
print("\n--- Answer ---")
print(response["output_text"])
print("---------------")
