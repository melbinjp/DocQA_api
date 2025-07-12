---
title: Gemini RAG Demo
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Gemini RAG Demo

A lightweight Retrieval-Augmented Generation (RAG) micro-service built with **FastAPI** and **Google Gemini Flash**.  
It ingests PDFs, DOCX, raw text or web URLs, creates MiniLM embeddings (in-memory), and answers questions in a single request.

---

## 🚀 Live on Hugging Face Spaces

This repo is ready to be pushed to [HF Spaces](https://huggingface.co/spaces). The Space is configured to run inside a tiny Docker container.

### Secrets
Add your **Google AI Studio** API key to the Space secrets: