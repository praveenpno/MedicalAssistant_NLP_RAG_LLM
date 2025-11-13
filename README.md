# RAG-based AI Medical Assistant for Healthcare Professionals

## Overview

This project addresses the critical challenges faced by healthcare professionals, such as information overload and the need for quick, accurate access to medical knowledge. It develops a Retrieval-Augmented Generation (RAG) based AI solution leveraging the extensive content of the Merck Manuals. The primary objective is to streamline decision-making, enhance diagnostic accuracy, and standardize care practices by providing a reliable and efficient way to query vast medical data. The solution aims to serve as a functional prototype demonstrating the feasibility and effectiveness of AI in supporting healthcare.

## Technologies & Libraries

The project utilizes a combination of robust AI and data processing libraries:

*   **Python**: The primary programming language.
*   **Jupyter Notebook**: For interactive development and execution.
*   **HuggingFace Hub**: For downloading pre-trained language models.
*   **llama-cpp-python**: Python bindings for LLaMA models, optimized for local execution with GPU support (CUDA/CUBLAS).
*   **LangChain**: A framework for developing applications powered by language models, used for text splitting, document loading, and retriever setup.
*   **PyMuPDF**: For loading and parsing PDF documents.
*   **ChromaDB**: A lightweight, open-source vector database for storing and querying document embeddings.
*   **Sentence Transformers (all-MiniLM-L6-v2)**: For generating dense vector embeddings of text.
*   **tiktoken**: For tokenization and managing text chunking.
*   **Pandas**: For data manipulation (though not extensively used in the provided content).
*   **Mistral-7B-Instruct-v0.2-GGUF**: The Large Language Model (LLM) employed for generating responses.

## Setup & Installation

To run this project, install the required dependencies:

```bash
pip install -r requirements.txt
```

And a suggested `requirements.txt`:

```
llama-cpp-python==0.1.85 # Ensure this version is compatible with your CUDA setup if using GPU
huggingface_hub
pandas
tiktoken
pymupdf
langchain
langchain-community
chromadb
sentence-transformers
numpy
```

**Note on `llama-cpp-python`**: For optimal GPU performance, it was installed with `CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.85 --force-reinstall --no-cache-dir -q`. Users with CUDA-enabled GPUs should follow a similar approach if performance is critical.

## How to Run

1.  Ensure you have a Python environment set up with the specified dependencies.
2.  Download the `Mistral-7B-Instruct-v0.2-GGUF` model file (`mistral-7b-instruct-v0.2.Q6_K.gguf`) from HuggingFace Hub as configured in the notebook.
3.  Place the `medical_diagnosis_manual.pdf` (Merck Manual) file in the specified path or adjust the path in the notebook.
4.  Open the `notebook.ipynb` file in a Jupyter environment (like Jupyter Lab, Jupyter Notebook, or VS Code with the Jupyter extension) and execute the cells sequentially.
5.  If running in Google Colab, ensure Google Drive is mounted to access the PDF file.

## Key Findings & Conclusion

The project demonstrates the progressive improvement in AI-driven medical information retrieval through different methodologies:

1.  **Initial LLM Responses**: Without specific grounding, the Large Language Model (LLM) provided general, often incomplete, and sometimes truncated answers. While it had general medical knowledge, its responses lacked the specificity and completeness required for clinical decision-making.
2.  **Prompt Engineering**: Implementing a system message and structured prompts improved the LLM's behavior, making it more cautious and capable of acknowledging when information wasn't directly available in a provided (though empty) context. However, it still relied on the LLM's general training data, leading to answers that were more structured but not necessarily more medically precise for complex queries.
3.  **Retrieval-Augmented Generation (RAG) with Fine-tuning**: This approach yielded significantly superior results. By integrating the Merck Manual as an external knowledge base, the RAG system could retrieve highly relevant document chunks and generate answers that were:
    *   **Accurate and Grounded**: Responses directly cited or accurately summarized information from the medical manual, as confirmed by an LLM-as-a-judge evaluation which consistently rated groundedness highly (mostly 5s).
    *   **Comprehensive and Relevant**: The RAG system provided detailed, comprehensive answers to medical queries, addressing all aspects of the user's question. The LLM-as-a-judge also affirmed high relevance scores (mostly 5s).
    *   **Actionable**: The detailed nature of the responses makes them more directly useful for healthcare professionals.

**Conclusion**: The RAG-based AI solution, leveraging high-quality medical manuals, is crucial for developing reliable and accurate medical assistants. It effectively overcomes the limitations of standalone LLMs by grounding their responses in verified, domain-specific knowledge. This approach significantly enhances the AI's ability to provide precise diagnostic assistance, drug information, treatment plans, and critical care protocols, making it a valuable tool to combat information overload and improve patient outcomes in healthcare settings. Continued investment in high-quality knowledge bases and user-friendly interfaces will be key to its successful deployment.