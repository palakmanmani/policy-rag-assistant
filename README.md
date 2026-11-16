# Insurance Policy Q&A Assistant (RAG)

A Python app that answers questions about an insurance policy PDF using
Retrieval-Augmented Generation (RAG) and the Gemini API. Every answer cites the
page number, and the app says "I could not find this in the policy" when the
policy does not contain the answer.

## Why I built this
An LLM does not know the contents of a private policy document and may guess.
In insurance, a wrong answer can cause a rejected claim. RAG fixes this by
giving the LLM only the relevant policy text.

## How it works
1. Read the PDF page by page (pypdf)
2. Split pages into 800-character chunks with 150 overlap
3. Convert chunks to embeddings (sentence-transformers, all-MiniLM-L6-v2)
4. For a question, find the 4 most similar chunks (cosine similarity)
5. Send those chunks and the question to Gemini with the rule
   "answer only from the excerpts"

## Tech stack
Python, pypdf, sentence-transformers, NumPy, Google Gemini API

## How to run
1. Clone the repo and create a virtual environment
2. `pip install -r requirements.txt`
3. Create a `.env` file with `GEMINI_API_KEY=your_key`
4. Put a policy PDF at `data/policy.pdf`
5. Run `python rag.py`

## Example
Q: What is the waiting period for pre-existing diseases?
A: 24 months (Page 21)

Q: Does this policy cover space travel?
A: I could not find this in the policy.

## Results
(I will add my accuracy numbers here after testing.)

## Limitations and next steps
- Works best on text PDFs, not scanned images
- Tables in the PDF can split badly across chunks
- Next: web UI with Streamlit, and an automatic accuracy test