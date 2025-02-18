import time
import faiss
import openai
import tiktoken
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

# Set your OpenAI API key from the 
openai.api_key = os.getenv("OPENAI_API_KEY")
print("key", openai.api_key)
if not openai.api_key:
    print("Warning: OPENAI_API_KEY is not set. Check your .env file.")

print()
# Global flag to avoid printing the same error repeatedly
_already_logged_openai_error = False

# Funzione per caricare il testo
def carica_testo(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Funzione per spezzare il testo in chunk
def suddividi_testo(text, chunk_size=500):
    parole = text.split()
    chunks = [' '.join(parole[i:i+chunk_size]) for i in range(0, len(parole), chunk_size)]
    return chunks

# Funzione per generare gli embedding OpenAI, with error handling and delay.
def get_embedding(text):
    global _already_logged_openai_error
    try:
        response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
        # Add a small delay to avoid rate limiting
        time.sleep(0.2)  # adjust delay as needed
        return np.array(response.data[0].embedding)
    except Exception as e:
        if not _already_logged_openai_error:
            print("Error calling OpenAI API for text:", text, "\nError:", e)
            _already_logged_openai_error = True
        # Even if there's an error, wait a bit before continuing to slow down the requests
        time.sleep(0.2)
        # Fallback: return a zero-vector of dimension 1536 (the expected output size for ada-002)
        return np.zeros(1536)

# Funzione per creare il database FAISS
def crea_faiss_db(chunks):
    dimension = len(get_embedding("test"))
    index = faiss.IndexFlatL2(dimension)
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        embeddings.append(emb)

    embeddings = np.array(embeddings, dtype=np.float32)
    index.add(embeddings)

    return index, chunks

# Funzione per cercare nel database FAISS
def cerca_in_faiss(query, index, chunks, top_k=3):
    query_emb = get_embedding(query).reshape(1, -1)
    _, indices = index.search(query_emb, top_k)
    risultati = [chunks[i] for i in indices[0]]
    return risultati

# Funzione per interrogare OpenAI con il contesto recuperato
def genera_risposta(query, index, chunks):
    contesto = cerca_in_faiss(query, index, chunks)
    prompt = f"Contesto:\n{contesto}\n\nDomanda: {query}\nRispondi dettagliatamente basandoti sul contesto sopra."
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Sei un assistente utile."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        # Optionally add a delay after the chat call as well
        time.sleep(0.2)
        return response.choices[0].message.content
    except Exception as e:
        print("Error generating answer with OpenAI:", e)
        return "Si è verificato un errore durante la generazione della risposta."

# At the end of extract_gpt.py, wrap the example/test code:
if __name__ == "__main__":
    file_path = "docs\\nuovo_grecita_short_pdf_testo_estratto.txt"
    testo = carica_testo(file_path)
    chunks = suddividi_testo(testo)
    index, chunks = crea_faiss_db(chunks)
    query = "che cos'è l'esame dattilico catalettico e riporta tutti i pareri elencati nel testo."
    risposta = genera_risposta(query, index, chunks)
    
    print("Risposta generata:", risposta)
    print("fine")

