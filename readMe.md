# RAG Model for PDF Querying

This project implements a Retrieval-Augmented Generation (RAG) model that allows you to query your PDF files and get answers based on their content.

## Features

- Upload PDF files as input.
- Ask questions related to the content of the PDFs.
- Get accurate and context-aware answers.

## Requirements

- Python 3.8+
- Required libraries:
  - `langchain`
  - `pypdf`
  - `openai`
  - `faiss` or `chromadb`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your PDF files in the `pdfs/` directory.
2. Run the script:
   ```bash
   python main.py
   ```
3. Follow the prompts to upload your PDF and ask questions.

## How It Works

1. **PDF Parsing**: Extracts text from the uploaded PDF files.
2. **Chunking**: Splits the text into smaller, manageable chunks.
3. **Embedding**: Converts text chunks into vector embeddings using a pre-trained model.
4. **Retrieval**: Finds the most relevant chunks for your query.
5. **Generation**: Uses a language model to generate answers based on the retrieved chunks.

## Example

```plaintext
> Upload your PDF: example.pdf
> Ask a question: What is the main topic of the document?
> Answer: The document discusses the implementation of AI models in healthcare.
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- OpenAI GPT models
- FAISS for vector similarity search
