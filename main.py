import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import torch
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os

# --- 1. Inisialisasi Aplikasi ---
app = FastAPI(title="SHSD RAG API", description="API untuk pencarian Kode Rekening Belanja")

# --- 2. Setup Model & Database (Dijalankan sekali saat startup) ---
print("⏳ Sedang memuat model dan database... Mohon tunggu.")

# Load CSV & Vector DB (Bisa di-load dari disk atau create baru)
# Disini kita create baru dari CSV agar aman di VPS
filename = "kode belanja v1.csv"
if os.path.exists(filename):
    df = pd.read_csv(filename)
    df['text_context'] = "Kode Rekening: " + df['REKENING BARU'].astype(str) + \
                         " | Uraian: " + df['NAMA REKENING BELANJA DAERAH'].astype(str)
    df = df.dropna(subset=['text_context'])
    
    loader = DataFrameLoader(df, page_content_column="text_context")
    docs = loader.load()
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Simpan di memori sementara atau folder khusus
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory="./chroma_db_vps" 
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
else:
    raise FileNotFoundError("File CSV tidak ditemukan!")

# Load Model Mistral
# Catatan: Di VPS pastikan punya GPU (Nvidia) jika pakai bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.01,
    do_sample=True,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Setup Chain
template = """
[INST] Anda adalah Asisten Ahli Kode Rekening Belanja Daerah (SHSD).
Gunakan data referensi berikut untuk menjawab:
{context}

Pertanyaan User: {question}

Berikan jawaban dengan format:
1. Rekomendasi Kode: [Kode]
2. Nama Rekening: [Nama]
3. Penjelasan Singkat: (Kenapa kode ini cocok)
[/INST]
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

print("✅ Sistem Siap!")

# --- 3. Definisi API Endpoint ---

class QueryRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "alive", "message": "SHSD AI Service is Running"}

@app.post("/predict")
def predict(request: QueryRequest):
    try:
        response = rag_chain.invoke(request.text)
        return {"query": request.text, "result": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 4. Entry Point ---
if __name__ == "__main__":
    # Host 0.0.0.0 agar bisa diakses dari luar VPS
    uvicorn.run(app, host="0.0.0.0", port=8000)