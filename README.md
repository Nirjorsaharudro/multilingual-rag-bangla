📚 multilingual-rag-bangla

This project implements a Retrieval-Augmented Generation (RAG) pipeline that supports both English and Bangla queries. It retrieves context from the HSC 2026 Bangla 1st Paper textbook and generates meaningful, grounded answers based on the retrieved content.
🛠️ Setup Instructions
1. Activate the Poetry Environment

poetry shell

2. Install Dependencies

poetry install

3. Set Your API Key

Create a .env file in the project root and add your OpenAI API key:

OPENAI_API_KEY=your_openai_api_key_here

4. Install Bangla Support for Tesseract OCR

sudo apt install tesseract-ocr-ben

5. Create the Vector Database

python create_vectorDB.py

6. Start the Backend

python main.py

7. Run the Frontend

Simply open index.html in your web browser.
No server setup is required for the frontend.
🎥 Demo
<video width="800" controls> <source src="Rag.mp4" type="video/mp4"> Your browser does not support the video tag. </video> 