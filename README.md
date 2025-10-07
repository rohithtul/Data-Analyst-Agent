How to Run:

python -m venv venv-clean

venv-clean\Scripts\activate

pip install streamlit pandas matplotlib langchain-google-genai google-generativeai langchain
windows Powershell:
$env:GEMINI_API_KEY="YOUR_API_KEY_HERE"

$env:GEMINI_MODEL = 'models/gemini-2.5-flash'  # updated based on preference

streamlit run app.py
<img width="1460" height="1121" alt="image" src="https://github.com/user-attachments/assets/c9eaa420-6887-4623-9229-1836c1fabdd9" />
