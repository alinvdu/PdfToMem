# 🧠 PdfToMem  
**Turn PDFs into structured, queryable memory—built for LLMs.**

Large Language Models struggle with memory. `PdfToMem` makes it effortless.  
By combining reasoning-powered ingestion, structured retrieval, and a multi-agent architecture, it transforms unstructured PDFs into rich memory representations.  

It exposes a powerful **MCP Server** to coordinate ingestion, reasoning, storage, and querying—using state-of-the-art tools like **LlamaIndex** and **LangGraph**.

---

## 🎥 Video Walkthrough
![Video Icon](https://github.com/user-attachments/assets/d8e0adb1-c455-43e7-9b7a-b01d14581042)[Watch the video](https://example.com/video.mp4)

## 🧠 MCP Server — _Memory Control Plane_

The **MCP Server** is the brain of PdfToMem. It leverages tool-based reasoning to:

- 🏗️ **Design ingestion pipelines**
- 🧩 **Choose the right memory representation**
- 📊 **Index and structure PDF content for retrieval**

Use tools like `determine_memory_architecture` to automatically infer the optimal memory structure using **LlamaIndex abstractions**:
- `VectorIndex`
- `QueryEngine`
- `LLMSelector`

---

## 🛠️ Tool-Driven Parsing — Powered by LangGraph

`PdfToMem` employs modular **tools** to extract structured data from PDFs:

- 📄 **Text Extraction**
- 📊 **Table Detection**
- 🧾 **OCR for Scanned Docs**
- 🖼️ **Image Extraction**
- 📸 **PDF Screenshots**

These tools feed into the **multi-agent system**, enabling intelligent, context-aware processing.

---

## 🤖 Multi-Agent Architecture

PdfToMem uses a **LangGraph-based orchestrator** to coordinate specialized agents. Each agent performs a distinct task:

- 🕵️ **Extractor Agent** – pulls raw content  
- 🧠 **Semantic Agent** – applies embedding & understanding  
- ✂️ **Segmenter Agent** – splits content intelligently  
- 🕸️ **Relationship Agent** – builds semantic links

The orchestrator decides **when and how** to invoke these agents based on the content type and reasoning goals.

---

## 🧭 Memory Planning — Reasoning Over Representation

Once the data is structured, a **planning model** (currently `o3-mini`) determines the **best memory format** using **LlamaIndex** components:

- 🧱 `SimpleSentenceIndex`  
- 🌐 `AutoMergingIndex`  
- 🧮 Custom tool-enhanced indices

This reasoning-driven approach ensures **optimal retrieval performance** tailored to the document.

---

## 💾 Storage — Flexible, Dynamic, Queryable

Based on the selected memory representation, a FastAPI-powered **Storage Service** builds a tailored **query engine** for each PDF.

- 🔍 Built-in search and retrieval
- 🧠 Vector and hybrid index support
- 🧪 Modular for experimentation

The storage is designed to support **scalable memory systems** across domains.

---

## 🖥️ MCP Client — Full Control with a Friendly UI

The **MCP Client** is a React-based interface for controlling the full lifecycle:

- ⚙️ Configure agents, tools, memory plans  
- 📂 Upload and ingest PDFs  
- 🔎 Preview structured chunks  
- 🧪 Query memory and view responses  

Everything is interactive, inspectable, and customizable.

---

## 🚀 Why PdfToMem?

- ✅ **LLM-optimized ingestion & memory**  
- 🧩 **Modular tools & agents**  
- 🧠 **Reasoning-based memory planning**  
- 💬 **Queryable representations via LlamaIndex**  
- 🌐 **UI + API for full pipeline control**

---

> 🛠️ _Contributions welcome! Help us build the future of intelligent memory systems._  
> 🔗 [MIT License](./LICENSE)
