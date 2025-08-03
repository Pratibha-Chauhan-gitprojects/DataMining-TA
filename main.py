from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from utils import extract_text_from_image, decode_base64_image
import faiss
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# Initialize FastAPI app
app = FastAPI()

# Load embedding model and FAISS index
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index/index.faiss")

# Load metadata and chunks
metadata_path = "faiss_index/metadata.txt"
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = [line.strip() for line in f]

# Load markdown chunks in memory to avoid 'chunks' not defined
chunks = []
data_dir = "markdown_data"
for fname in os.listdir(data_dir):
    if fname.endswith(".md"):
        with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
            content = f.read().split("## [Page")
            for section in content[1:]:
                chunk = section.split("]", 1)[1].strip()
                if len(chunk) > 20:
                    chunks.append(chunk)

# Input model
class Query(BaseModel):
    query: str = ""
    image_base64: str | None = None

@app.post("/ask")
def ask(query_obj: Query):
    query = query_obj.query.strip()

    # Append image text if provided
    if query_obj.image_base64 and is_base64_image(query_obj.image_base64):
        try:
            query += "\n" + extract_text_from_image(query_obj.image_base64)
        except Exception as e:
            return {"error": f"Failed to extract image text: {str(e)}"}

    # Embed and search in FAISS
    embedding = embedder.encode([query])
    D, I = index.search(embedding, k=5)
    sources = [metadata[i].strip() for i in I[0]]

    # Build textbook context
    top_chunks = []
    for idx in I[0]:
        page_info = metadata[idx]
        page_number = page_info.split(":")[-1] if ":" in page_info else page_info
        top_chunks.append(f"[Page {page_number}]\n{chunks[idx]}")
    book_context = "\n\n---\n\n".join(top_chunks)

    # Step 1: Ask Gemini to simulate an internet search
    try:
        search_prompt = f"""You are an AI search assistant.
Search the web and summarize the best available answer to this question:
"{query}"

Respond with links and brief supporting text only. No extra commentary.
3. Search online for additional answers. Share results WITH CITATION LINKS.
4. Think step-by-step. Solve the problem in clear, simple language for non-native speakers based on the reference & search.
5. Follow-up: Ask thoughtful questions to help students explore and learn.
"""
        search_response = model.generate_content(search_prompt)
        web_results = search_response.text.strip()
    except Exception as e:
        web_results = "No additional web results could be found due to error: " + str(e)

    # Step 2: Combine both into a final answer prompt
    final_prompt = f"""You are a Teaching Assistant (TA) for the Data Mining Course.
Below is context retrieved from textbooks and course materials. Start by using this context to answer the question in detail. 
Use markdown formatting. Cite the source page (given in brackets at the start of each section) wherever textbook content is used. DO not add textbook name in answer part only page number but add both book name and page number in source.

### Book Context:
{book_context}

### Web Search Results:
{web_results}

### Question:
{query}

### Answer (Markdown):
"""

    try:
        final_response = model.generate_content(final_prompt)
        return {
            "answer": final_response.text.strip(),
            "sources": sources,
            "web_summary": web_results
        }
    except Exception as e:
        return {"error": str(e)}
