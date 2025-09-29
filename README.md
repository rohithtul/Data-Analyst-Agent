How to Run:

python -m venv venv-clean

venv-clean\Scripts\activate

pip install streamlit pandas matplotlib langchain-google-genai google-generativeai langchain

$env:GEMINI_MODEL = 'models/gemini-2.5-flash'  # updated based on preference

streamlit run app.py
