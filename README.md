# 🤖 Intelligent HR Helpdesk Chatbot (RAG-Powered)

An automated, intelligent HR Helpdesk assistant designed to alleviate repetitive policy inquiries for **Zyro Dynamics Pvt Ltd** (a fast-growing SaaS company with 500+ employees). 

This system utilizes a **Retrieval-Augmented Generation (RAG)** pipeline to query internal company documentation securely and accurately. It features an evaluation guardrail layer (`Guardials`) to reject out-of-scope inquiries and uses an ultra-low latency inference engine to deliver rapid human-centric answers via a clean web UI.

---

## 🛠️ Tech Stack Architecture

| Layer | Component / Technology | Exact Package / Identifier | Primary Functionality |
| :--- | :--- | :--- | :--- |
| **Frontend UI** | Streamlit | `streamlit` | Handles interactive web layout and real-time session state management. |
| **Orchestration** | LangChain | `langchain-core` / `langchain-community` | Chains prompt workflows, text splitters, and models together via LCEL. |
| **Data Ingestion**| PyPDF | `pypdf` | Extracts raw textual metadata out of company internal policy PDFs. |
| **Text Chunking** | LangChain Splitters | `langchain-text-splitters` | Implements `RecursiveCharacterTextSplitter` (Chunk: 800, Overlap: 100). |
| **Embeddings** | Hugging Face | `sentence-transformers/all-MiniLM-L6-v2` | Converts textual chunks into dense numerical vector mappings locally. |
| **Vector Index** | FAISS (Meta) | `faiss-cpu` | Efficient in-memory index for rapid similarity calculations. |
| **Inference API** | Groq Cloud | `langchain-groq` | Manages high-speed connections to open-source LLM infrastructures. |
| **Foundation LLM**| GPT-OSS | `openai/gpt-oss-20b` | Powering the scope categorization filter and natural language responses. |
| **Config/Security**| Dotenv | `python-dotenv` | Injects sensitive runtime variables (`GROQ_API_KEY`) into memory. |

---

## 📂 Project Structure

```text
Intelligent RAG/
│
├── .venv/                          # Python Virtual Environment
├── zyro-dynamics-hr-corpus/        # Knowledge Base folder hosting internal policy PDFs
│   ├── leave_policy.pdf
│   ├── reimbursement_guidelines.pdf
│   └── code_of_conduct.pdf
│
├── app.py                          # Main Streamlit UI & Integrated RAG script
├── code.ipynb                      # Development and prototyping notebook
├── requirements.txt                # List of system package dependencies
└── .env                            # Secure environment configuration variables
```

---

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com
cd intelligent-hr-rag
```

### 2. Configure Environment Variables
Create a file named `.env` in the root directory and append your high-speed Groq access token:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### 3. Initialize Virtual Environment & Install Dependencies
Avoid Windows environment binary execution path faults by launching modules directly through Python:
```bash
# Create Virtual Environment
python -m venv .venv

# Activate Virtual Environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install Project Dependencies
python -m pip install -r requirements.txt
```

---

## 🚀 Running the Application

Once your internal policy documents are placed inside the `zyro-dynamics-hr-corpus/` directory, spin up the local development web server:

```bash
streamlit run app.py
```

The application will run optimizations automatically, ingest files, build your FAISS index matrix, and open a browser window displaying the chatbot dashboard at `http://localhost:8501`.

---

## 🛡️ Core System Features

* **Intent Guardrails (`Guardials`):** Dynamically pre-screens inputs. If an employee submits an unrelated text request (e.g., coding assistance or general history questions), the system safely intercepts it as `OUT_SCOPE` without executing document vector storage calls.
* **Smart Resource Caching:** Uses `@st.cache_resource` flags so heavy deep-learning embedding layers and source documents are built into vector indices **once** upon boot, ensuring rapid conversation iterations.
* **Context-Driven Answers:** System strictly checks target arrays using `RAG_PROMPT`. If information is missing from internal corpus sources, the agent yields to default constraints rather than hallucinating answers.
