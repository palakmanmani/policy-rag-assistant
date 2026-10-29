import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()  # reads your key from the .env file
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-flash-latest"  # <- change this to the model name you saw in AI Studio

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        pages.append((i + 1, page.extract_text() or ""))
    return pages


def chunk_pages(pages, size=800, overlap=150):
    chunks = []
    for page_no, text in pages:
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({"page": page_no, "text": piece})
            start += size - overlap
    return chunks


def build_index(chunks):
    texts = [c["text"] for c in chunks]
    return embedder.encode(texts, normalize_embeddings=True)


def retrieve(question, chunks, vectors, k=4):
    q = embedder.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ q
    top = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top]


def answer(question, chunks, vectors):
    results = retrieve(question, chunks, vectors)
    context = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c, _ in results)
    prompt = f"""You are an insurance policy assistant.
Answer the question using ONLY the policy excerpts below.
Mention the page number(s) you used.
If the answer is not in the excerpts, say exactly: "I could not find this in the policy."

Policy excerpts:
{context}

Question: {question}
"""
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text


if __name__ == "__main__":
    pages = load_pdf("data/policy.pdf")
    chunks = chunk_pages(pages)
    print("Loading... please wait")
    vectors = build_index(chunks)
    print("Ready!")

    while True:
        q = input("\nAsk a question (or type quit): ")
        if q.lower() == "quit":
            break
        print("\n" + answer(q, chunks, vectors))