🤖 RAG based Document & Website Chatbot



This project is a genAI based versatile Retrieval-Augmented Generation (RAG) application that allows users to have a conversation with custom knowledge bases. It can ingest data from multiple sources, including public websites, uploaded PDF documents, and entire Google Drive folders.

The entire application is wrapped in a user-friendly web interface built with Streamlit, allowing users to select their data source and get answers in real-time.

✨ Features
Multi-Source Ingestion: Ingest data by providing a public website URL, uploading local PDF files, or specifying a Google Drive folder ID.

Vector-Based Retrieval: Uses Google's state-of-the-art embedding models to create vector representations of the document's text.

Efficient Semantic Search: Stores text vectors in a ChromaDB vector database for fast and relevant context retrieval.

Context-Aware Answers: Integrates with the Google Gemini API to generate accurate, natural-language answers based only on the information found in the source document.

Interactive UI: A simple and intuitive web interface built with Streamlit allows for easy interaction without needing to touch the code.

Caching: Caches processed documents (URLs, PDFs, and folders) to avoid re-loading and re-processing the same source, leading to a much faster user experience.

🛠️ Tech Stack
Language: Python

Core Framework: LangChain

LLM & Embeddings: Google Gemini API

Vector Database: ChromaDB

Web UI: Streamlit

Data Loaders: BeautifulSoup4 (Websites), PyMuPDF (PDFs), Google Drive API

⚙️ Setup and Installation
Follow these steps to set up and run the project locally on your machine.

Prerequisites
Python 3.9+

Conda (or another virtual environment manager like venv)

Installation Guide

Clone the Repository:

""
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
""


Create and Activate a Virtual Environment:

""
conda create --name rag_chatbot python=3.11
conda activate rag_chatbot
""


Install Dependencies:
This project's dependencies are listed in the requirements.txt file. Install them all with a single command:

""
pip install -r requirements.txt
""


Configure API Key:
The application uses a secrets.toml file to manage your Google API key securely.

In the root of your project folder, create a new folder named .streamlit.

Inside the .streamlit folder, create a new file named secrets.toml. ( The path will be '.streamlit/secrets.toml')

Add your Google Gemini API key to this file:

# .streamlit/secrets.toml

""
GOOGLE_API_KEY = "YOUR_GOOGLE_AI_API_KEY"  
""
(paste your api key inside "YOUR_GOOGLE_AI_API_KEY")



(For Google Drive Only) Configure Drive API Access:

Follow the Google Cloud Console setup instructions to enable the Drive API and create OAuth 2.0 credentials for a Desktop App.

Download the credentials file and rename it to credentials.json.

Place this credentials.json file in the root of your project folder (the same directory as app.py).

▶️ How to Run
To launch the web application, run the following command from the root of your project folder:

Bash

streamlit run app.py
Your default web browser will automatically open a new tab with the running application. You can then:

Select your desired data source ("Website URL", "Upload PDF", or "Google Drive Folder").

Provide the input.
(For google drive, input the drive ID, eg-https://drive.google.com/drive/u/1/folders/ufhqfiuhqeuifhogf4ghgho875gh34241, here ufhqfiuhqeuifhogf4ghgho875gh34241 
is the drive ID)

Start asking questions about the loaded content.

Note: The first time you use the Google Drive feature, your browser will open a new tab for you to log in and authorize the application to access your files.
