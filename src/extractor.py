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

if __name__ == "__main__":
    percorso_pdf = "docs/nuovo_grecita_short.pdf"  # Replace with your PDF path
    testo_estratto = estrai_testo_da_pdf(percorso_pdf)
    print(testo_estratto)
    
    # Write the extracted text to a file
    with open(f"{percorso_pdf.replace('.', '_')}_testo_estratto.txt", "w", encoding="utf-8") as file_output:
        file_output.write(testo_estratto)
