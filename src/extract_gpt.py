import faiss
import openai
import tiktoken
import numpy as np

# Imposta la tua API key di OpenAI
openai.api_key = "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"

# Funzione per caricare il testo
def carica_testo(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Funzione per spezzare il testo in chunk
def suddividi_testo(text, chunk_size=500):
    parole = text.split()
    chunks = [' '.join(parole[i:i+chunk_size]) for i in range(0, len(parole), chunk_size)]
    return chunks

# Funzione per generare gli embedding OpenAI
def get_embedding(text):
    response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
    return np.array(response.data[0].embedding)

# Funzione per creare il database FAISS
def crea_faiss_db(chunks):
    dimension = len(get_embedding("test"))  # Otteniamo la dimensione degli embedding
    index = faiss.IndexFlatL2(dimension)  # Creiamo un index FAISS

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

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Sei un assistente utile."},
                  {"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# 1. Carica il testo salvato
file_path = "docs\\nuovo_grecita_short_pdf_testo_estratto.txt"
testo = carica_testo(file_path)

# 2. Suddividi il testo in chunk
chunks = suddividi_testo(testo)

# 3. Crea il database FAISS
index, chunks = crea_faiss_db(chunks)

# 4. Esempio di query all'IA
query = "che cos'è l'esame dattilico catalettico e riporta tutti i pareri elencati nel testo."
risposta = genera_risposta(query, index, chunks)

print("Risposta generata:", risposta)
print("fine")
