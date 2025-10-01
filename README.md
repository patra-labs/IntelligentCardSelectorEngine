# Credit Card RAG Recommender (Version 1)

This project is a **RAG (Retrieval-Augmented Generation) system** to recommend the best credit card based on user queries. It leverages official credit card **benefit documents** (PDFs) and uses **OpenAI embeddings + GPT models** for reasoning and recommendation.

---

## 🔹 Features

* Reads credit card **benefit guides (PDFs)**.
* Stores benefit text in a **vector database (ChromaDB)** for semantic retrieval.
* Uses **OpenAI embeddings** to find relevant information based on user queries.
* Provides **card recommendations** using GPT with explanations.
* Modular structure for **easy extension** in future versions (Version 2 will include structured fields, scoring, and advanced filtering).

---