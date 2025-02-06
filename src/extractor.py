import PyPDF2

def estrai_testo_da_pdf(percorso_pdf):
    try:
        with open(percorso_pdf, "rb") as file:
            lettore = PyPDF2.PdfReader(file)
            testo_completo = ""
            for pagina in range(len(lettore.pages)):
                testo_completo += lettore.pages[pagina].extract_text() + "\n"
            return testo_completo
    except Exception as e:
        return f"Errore nell'estrazione del testo: {str(e)}"

# Esempio di utilizzo
percorso_pdf = "docs/nuovo_grecita_short.pdf"  # Sostituisci con il percorso del tuo PDF
testo_estratto = estrai_testo_da_pdf(percorso_pdf)
print(testo_estratto)

# Stampare il testo estratto
with open(f"{percorso_pdf.replace('.', '_')}_testo_estratto.txt", "w", encoding="utf-8") as file_output:
    file_output.write(testo_estratto)
