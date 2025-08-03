import os
import fitz  # PyMuPDF
from markdownify import markdownify as md

input_dir = "reference_pdfs"
output_dir = "markdown_data"
os.makedirs(output_dir, exist_ok=True)

def extract_text_with_pages(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            header = f"\n\n## [Page {i+1}] — {os.path.basename(pdf_path)}\n\n"
            texts.append(header + text.strip())
    return "\n".join(texts)

for filename in os.listdir(input_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(input_dir, filename)
        text = extract_text_with_pages(pdf_path)
        md_text = md(text)
        out_path = os.path.join(output_dir, filename.replace(".pdf", ".md"))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"✅ Saved: {out_path}")
