

# 🧠 Virtual Patient AI Training System

This project provides a local deployment of the Virtual Patient AI Training System, enabling model management, data processing, fine-tuning, and evaluation—all through an interactive Streamlit interface.

## 🚀 Prerequisites

Ensure the following are installed on your machine:

- Python 3.11 or higher  
- Git (for cloning the repository)

## 📥 Step 1: Download the Code

Clone or download the entire project, including all Python files and directory structure:

```bash
git clone https://github.com/your-repo/virtual-patient-ai.git
cd virtual-patient-ai
```

> Replace the URL with your actual repository link.

## 🧪 Step 2: Install Python Dependencies

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:

```bash
pip install streamlit pandas torch transformers peft datasets numpy requests pathlib2
```

## 🦙 Step 3: Install and Setup Ollama

Install Ollama:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Start the Ollama service:

```bash
ollama serve
```

In a new terminal, pull a model:

```bash
ollama pull gemma:2b
# or
ollama pull llama2:7b
```

## ▶️ Step 4: Run the Application

Navigate to your project directory and launch the Streamlit app:

```bash
cd /path/to/your/project
streamlit run app.py --server.port 8501
```

## 🌐 Step 5: Access the Application

Open your browser and visit:

```
http://localhost:8501
```

## 📁 Directory Structure

Ensure your project follows this structure:

```
project/
├── app.py
├── src/
│   ├── model_manager.py
│   ├── fine_tuning.py
│   ├── persona_classifier.py
│   ├── data_processor.py
│   ├── evaluation.py
│   └── ollama_client.py
├── utils/
│   ├── config.py
│   └── helpers.py
└── .streamlit/
    └── config.toml
```

## 🛠️ Troubleshooting

- **Ollama "not connected" error**: Ensure `ollama serve` is running in the background.
- **Dependency issues**: Try installing packages individually:
  ```bash
  pip install streamlit
  pip install torch
  # ...and so on
  ```
- **GPU support**: Install PyTorch with CUDA if you have an NVIDIA GPU.

---

