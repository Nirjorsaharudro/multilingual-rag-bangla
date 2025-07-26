# multilingual-rag-bangla
This project implements a Retrieval-Augmented Generation (RAG) pipeline that supports both English and Bangla queries. It retrieves context from the HSC26 Bangla 1st Paper textbook and generates meaningful, grounded answers.


Setup Instructions

1. First, activate the Poetry environment:

poetry shell

2. Then, install the required dependencies:

poetry install

3. Create a .env file in the project root and add your OpenAI API key:

OPENAI_API_KEY=your_openai_api_key_here

4. Install Bangla language support for Tesseract OCR:

sudo apt install tesseract-ocr-ben

5. Create the vector database by running:

python create_vectorDB.py

6. Start the backend with:

python main.py

7. For the frontend, simply open index.html in a web browser. No server is needed for this.