import os
from fastapi import FastAPI

app = FastAPI()

# Ottieni la porta dalla variabile d'ambiente, se disponibile
port = int(os.environ.get("PORT", 3000))

# Avvia il server sulla porta dinamica
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

