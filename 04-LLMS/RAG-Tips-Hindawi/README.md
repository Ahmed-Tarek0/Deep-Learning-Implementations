# Simple RAG System — Tips Hindawi University

This project was built as an assignment for the **Tips Hindawi internship**. The task was to
build a simple Retrieval-Augmented Generation (RAG) system that answers questions about
Tips Hindawi University using a provided PDF as the only knowledge source.

## What this project does

Given `Tips_Hindawi_University_Info.pdf` (a fact sheet about a fictional university used for
this assignment), the notebook builds a full RAG pipeline that:

1. Loads and reads the PDF.
2. Splits it into overlapping text chunks.
3. Embeds every chunk with a Hugging Face sentence embedding model.
4. Retrieves the most relevant chunks for a given question using cosine similarity.
5. Feeds those chunks to a Hugging Face generative model to produce a short, grounded answer.
6. Falls back to `"Not mentioned in the document."` when the retrieved chunks aren't relevant
   enough, instead of guessing.

Questions are entered interactively — the notebook prompts you to type a question and answers
it on the spot, until you type `exit` or `quit`.

## Pipeline

```
PDF → raw text → chunks (60 words, 15-word overlap) → embeddings (computed once)
                                                              ↓
your question → embedding → cosine similarity against every chunk
                                                              ↓
                     filter by similarity threshold + take top_k
                                                              ↓
            no chunk passes → "Not mentioned in the document."
            chunk(s) pass   → prompt + generative model → answer
```

## Models used

| Component  | Model                                             | Why                                                                 |
|------------|----------------------------------------------------|----------------------------------------------------------------------|
| Embedder   | `sentence-transformers/all-MiniLM-L6-v2`           | Small (~80MB), fast on CPU, standard choice for semantic search.    |
| Generator  | `google/flan-t5-large`                              | Instruction-tuned; follows "answer in your own words, don't copy the context" reliably, unlike the smaller flan-t5 variants which tended to copy context verbatim. |

## Requirements

```
pypdf
sentence-transformers
transformers
torch
numpy
```

Install with:

```bash
pip install pypdf sentence-transformers transformers torch numpy
```

(The notebook also installs these itself in its first cell.)

## How to run

1. Open `RAG_System_Notebook.ipynb` in Google Colab (recommended) or Jupyter.
2. Make sure `Tips_Hindawi_University_Info.pdf` is in the same working directory as the notebook
   (upload it in Colab if needed).
3. Run all cells in order (**Runtime → Restart and run all** in Colab).
4. The last cell will prompt you to type a question. Type `exit` or `quit` to stop.

## Notes / limitations

- This is a from-scratch, dependency-light RAG implementation built for a learning assignment —
  not a production system.
- The generator only answers from what the retriever finds in the PDF; it does not use outside
  knowledge.
- The similarity threshold (`min_score = 0.40`) was tuned by testing against the four sample
  questions below; it may need re-tuning for a different document.

## Example questions

- Where is Tips Hindawi University located?
- Does the university offer online programs?
- Is there financial aid for international students?
- What languages are used for instruction?
