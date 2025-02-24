import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from src.extract_gpt import carica_testo, suddividi_testo, crea_faiss_db, genera_risposta
from src.extractor import estrai_testo_da_pdf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


# Configura CORS per permettere richieste dal frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://didem.vercel.app"],  # Specifica l'origine del tuo frontend, es. ["https://didem.vercel.app/"]
    allow_credentials=True,
    allow_methods=["*"],  # Permette tutti i metodi (GET, POST, ecc.)
    allow_headers=["*"],  # Permette tutti gli headers
)


# Directory configuration
DOCS_DIR = "docs"
UPLOADS_DIR = os.path.join(DOCS_DIR, "uploads")

# Ensure upload directory exists.
os.makedirs(UPLOADS_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, save it under docs/uploads, extract its text,
    and save the extracted text to disk.
    """
    try:
        # Save uploaded file to the uploads folder.
        file_path = os.path.join(UPLOADS_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract text from the uploaded PDF.
        extracted_text = estrai_testo_da_pdf(file_path)
        
        # Save the extracted text to a new file.
        text_filename = file.filename.replace(".", "_") + "_testo_estratto.txt"
        text_file_path = os.path.join(UPLOADS_DIR, text_filename)
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        
        return {
            "message": "PDF uploaded and processed successfully.",
            "file": file.filename,
            "extracted_text": extracted_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-pdfs")
def list_pdfs():
    try:
        pdf_files = []
        # Walk through the DOCS_DIR recursively.
        for root, dirs, files in os.walk(DOCS_DIR):
            for f in files:
                if f.lower().endswith(".pdf"):
                    # Get a relative path from DOCS_DIR.
                    relative_path = os.path.relpath(os.path.join(root, f), start=DOCS_DIR)
                    pdf_files.append(relative_path)
        return {"pdf_files": pdf_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-query")
def process_query(query: str = Form(...), pdf_filename: str = Form(...)):
    """
    Process a user query on a specific PDF. The endpoint:
      1. Checks if the extracted text file exists for the given PDF.
      2. If not, extracts text from the PDF and saves it.
      3. Builds a FAISS index from the extracted text.
      4. Processes the query using GPT-4 and returns the answer.
    """
    try:
        # Construct file paths.
        pdf_path = os.path.join(UPLOADS_DIR, pdf_filename)
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail=f"PDF file '{pdf_filename}' not found.")
        
        text_filename = pdf_filename.replace(".", "_") + "_testo_estratto.txt"
        text_file_path = os.path.join(UPLOADS_DIR, text_filename)
        
        # If the extracted text file doesn't exist, extract and save.
        if not os.path.exists(text_file_path):
            extracted_text = estrai_testo_da_pdf(pdf_path)
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)
        else:
            extracted_text = carica_testo(text_file_path)
        
        # Build FAISS index.
        chunks = suddividi_testo(extracted_text)
        index, chunks = crea_faiss_db(chunks)
        
        # Generate answer.
        answer = genera_risposta(query, index, chunks)
        return {"pdf_filename": pdf_filename, "query": query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-all")
def query_all(query: str = Form(...)):
    """
    Process a user query across all PDFs in the docs folder.
    This endpoint:
      1. Aggregates all extracted text from files ending with "_testo_estratto.txt"
         in the DOCS_DIR (including uploads).
      2. Builds a FAISS index from the aggregated text.
      3. Processes the query using GPT-4 and returns the answer.
    """
    try:
        aggregated_text = ""
        # List all text files in DOCS_DIR and UPLOADS_DIR
        for directory in [DOCS_DIR, UPLOADS_DIR]:
            for file in os.listdir(directory):
                if file.endswith("_testo_estratto.txt"):
                    file_path = os.path.join(directory, file)
                    aggregated_text += carica_testo(file_path) + "\n"
        
        if not aggregated_text.strip():
            raise HTTPException(status_code=404, detail="No extracted text found. Please upload PDFs first.")
        
        # Build FAISS index from aggregated text.
        chunks = suddividi_testo(aggregated_text)
        index, chunks = crea_faiss_db(chunks)
        
        # Generate answer.
        answer = genera_risposta(query, index, chunks)
        return {"query": query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
