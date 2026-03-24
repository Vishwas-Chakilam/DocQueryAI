from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_test_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "DocQuery AI Test Document")
    
    c.setFont("Helvetica", 12)
    text = [
        "Project Secret Code: 42-XJ-9",
        "Lead Developer: Antigravity AI",
        "Project Launch Date: March 24, 2026",
        "",
        "Topic 1: Retrieval-Augmented Generation (RAG)",
        "RAG is a technique used to give large language models (LLMs) access to real-time data.",
        "It combines the power of vector databases with LLM generative capabilities.",
        "",
        "Topic 2: ChromaDB",
        "ChromaDB is a developer-friendly vector database used for semantic search.",
        "It supports persistence and can handle large volumes of documents efficiently.",
        "",
        "Final Fact: The meaning of life is 42.",
    ]
    
    y = height - 100
    for line in text:
        c.drawString(50, y, line)
        y -= 20
        
    c.save()

if __name__ == "__main__":
    create_test_pdf("test_document.pdf")
    print("test_document.pdf created successfully!")
