###1)
import PyPDF2###

### Function to extract from PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text###

### Path to the PDF file
pdf_path = r'C:\Users\lmesd\OneDrive\desktop\111.pdf'###

pdf_text = extract_text_from_pdf(pdf_path)###

### Output the text
print(pdf_text)###

###2)
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import PyPDF2###

def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text###

### Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')###

### Using BERT
def text_to_vector(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()###

### Create FAISS index
def create_faiss_index(vectors):
    index = faiss.IndexFlatL2(vectors.shape[1])  # Using L2 distance for the index
    index.add(np.array(vectors, dtype=np.float32))
    return index###

### Path to the PDF file
pdf_file_path = r'C:\Users\lmesd\OneDrive\桌面\111.pdf'###

### Extract text from PDF file
pdf_text = extract_text_from_pdf(pdf_file_path)###

document_vectors = text_to_vector(pdf_text)###

faiss_index = create_faiss_index(document_vectors)###

### Example
query = "data analysis"
query_vector = text_to_vector(query)###

### Perform the search
D, I = faiss_index.search(np.array(query_vector, dtype=np.float32), k=5)###

### Print the top 5 results
print(f"Top 5 results for the query '{query}': {I}")###




