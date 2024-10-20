import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from functools import cache
from typing import Iterator, Mapping, TypeVar
import torch
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain import hub
from dotenv import load_dotenv
import os

load_dotenv()  # This loads the .env file
api_key = os.getenv("LANGCHAIN_API_KEY")

# Title for the app
st.title("RAG System")

# Step 1: PDF Upload UI
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    
    # Step 2: Loading the PDF
    pdf_path = uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    # Step 3: Splitting the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)
    
    str_docs = []
    for i, doc in enumerate(docs):
        str_docs.append(str(docs[i]))

    # Step 4: Embedding generation and FAISS indexing
    @cache
    def load_embedding_model():
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model

    model = load_embedding_model()

    embeddings = model.encode(str_docs)
    d = embeddings.shape[1]
    
    # Create FAISS index
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    # Define the VectorMap class
    T = TypeVar("T")

    class VectorMap(Mapping[str, T]):
        def __init__(self, data, threshold: float = 0.7):
            self._data: list[T] = data
            self._model = model
            self._embeddings = self._model.encode(self._data, convert_to_tensor=True)
            self._length = len(data)
            self._threshold = threshold

        def __getitem__(self, key) -> T:
            out = []
            top_k = 4
            query_embedding = self._model.encode(key, convert_to_tensor=True)
            similarity_scores = self._model.similarity(query_embedding, self._embeddings)[0]
            scores, indices = torch.topk(similarity_scores, k=top_k)
            for i,(score, index) in enumerate(zip(scores, indices)):
                if score > self._threshold:
                    out.append(f'{i+1}- ' + self._data[index])
            return '\n'.join(out)

        def __contains__(self, key) -> bool:
            query_embedding = self._model.encode(key, convert_to_tensor=True)
            scores = self._model.similarity_scores(query_embedding, self._embeddings)
            return any(score > self._threshold for score in scores)

        def __iter__(self) -> Iterator[str]:
            return iter(self._data)

        def __len__(self):
            return self._length

        def __repr__(self):
            return "VectorMap()"

    vector_map = VectorMap(str_docs, threshold=0)

    # Step 5: LLM integration using Ollama
    llm = ChatOllama(
        model="llama3",
        temperature=0.4,
    )

    # Step 6: UI for Question Input
    question = st.text_input("Ask a question related to the PDF content:")

    # Step 7: Retrieval QA process
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    chain = prompt | llm | StrOutputParser()

    if st.button("Get Answer") and question:
        print('recieving done!')

        # Retrieve relevant context from the vector map
        context = vector_map[question]
        print(context)
        # Generate answer from LLM
        answer = chain.invoke(
                    {
                        "input": f"{question}",
                        "context": context,
                    }
                )
        print('answer done!')
        st.write(f"Answer: {answer}")
        print('UI done!')