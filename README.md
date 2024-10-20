# Retrieval-Augmented Generation (RAG) System

This project is a fully open-source Retrieval-Augmented Generation (RAG) system that combines document retrieval and question-answering capabilities using large language models (LLMs). It includes a Jupyter notebook (`RAG.ipynb`) for building the RAG system, as well as a Streamlit app (`app.py`) for user interaction.

## Features
- **Document Upload**: Users can upload a PDF document for content analysis.
- **Text Chunking**: The PDF content is split into smaller chunks for processing.
- **Embedding Generation**: The chunks are embedded using a sentence transformer model (`all-MiniLM-L6-v2`).
- **FAISS Indexing**: The embeddings are stored and indexed using FAISS to allow for fast similarity search.
- **Question Answering**: A large language model (LLM) is used to answer questions related to the document content, leveraging the context retrieved from the FAISS index.

## Technologies Used
- **LangChain**: For building and managing chains of language models.
- **Ollama**: For interacting with the Llama 3 language model.
- **Sentence Transformers**: To generate embeddings of text chunks.
- **FAISS**: To index and retrieve document embeddings.
- **Streamlit**: For building a web interface for uploading documents and asking questions.

## Installation

### Prerequisites
- Python 3.8+
- Install the following packages:
  - `langchain-ollama`
  - `langchain-community`
  - `sentence-transformers`
  - `faiss`
  - `streamlit`
  - `torch`
  - `python-dotenv`

### Setting Up the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/rag-system.git
   cd rag-system
   ```

2. **Install dependencies:**
   Run the following command to install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your `.env` file**:
   Create a `.env` file in the project root to store your API keys (if needed):
   ```
   LANGCHAIN_API_KEY=your_api_key_here
   ```

## Usage

### Running the Jupyter Notebook
1. Open the `RAG.ipynb` notebook in Jupyter.
2. Follow the code to initialize the language model, load the document, split it into chunks, and create the FAISS index.
3. Ask questions related to the document content using the Llama 3 language model, which provides answers based on retrieved context.

### Running the Streamlit App
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Upload a PDF file through the UI.
3. Ask a question related to the content of the uploaded document.
4. The system will retrieve relevant context from the document and generate an answer using the language model.

## How it Works

1. **PDF Processing**: The system first loads the PDF file and splits the text into manageable chunks.
2. **Embedding and Indexing**: The text chunks are embedded using a sentence transformer model, and a FAISS index is created to store these embeddings.
3. **Context Retrieval**: When a user asks a question, the system retrieves relevant chunks of text from the document by performing a similarity search using FAISS.
4. **LLM Answering**: The retrieved context is passed to the Llama 3 model, which generates a coherent and contextually accurate answer.

## Example

- You upload a PDF containing information on AI research.
- You ask: _"What are some potential ideas for a graduation project in AI?"_
- The system retrieves relevant sections from the document and uses the Llama 3 model to generate project ideas.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments
- [LangChain](https://www.langchain.com/)
- [Ollama](https://www.ollama.com/)
- [Sentence Transformers](https://huggingface.co/sentence-transformers)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
