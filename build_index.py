import os
import faiss
import openai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

data_dir = "markdown_data"
index_dir = "faiss_index"
os.makedirs(index_dir, exist_ok=True)

texts, metadata = [], []
for fname in os.listdir(data_dir):
    if fname.endswith(".md"):
        with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
            content = f.read().split("## [Page")
            for section in content[1:]:
                page = section.split("]")[0].strip()
                chunk = section.split("]", 1)[1].strip()
                if len(chunk) > 20:
                    texts.append(chunk)
                    metadata.append(f"{fname} | Page {page}")

embeddings = model.encode(texts)
index = faiss.IndexFlatL2(384)
index.add(embeddings)
faiss.write_index(index, os.path.join(index_dir, "index.faiss"))

with open(os.path.join(index_dir, "metadata.txt"), "w", encoding="utf-8") as f:
    for line in metadata:
        f.write(line + "\n")

print("✅ FAISS index built and saved.")
