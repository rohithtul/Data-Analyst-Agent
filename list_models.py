"""List available Google GenAI models for the configured API key.

Usage:
  - Activate your venv, ensure GEMINI_API_KEY is set, then run:
      python list_models.py

This script will try multiple methods to list models:
 1. Use the google-generative-ai Python client if installed.
 2. Fall back to making a REST call to the Google GenAI `models:list` endpoint using the API key.

Note: For some accounts/models you may need OAuth credentials or a service account; this script uses the API key approach which works for many simple setups.
"""

import os
import sys
import json
import urllib.request
import urllib.parse

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("GEMINI_API_KEY environment variable not set. Set it and re-run.")
    sys.exit(1)

# Try to use google generative ai client if available
try:
    from google.generativeai import client as genai_client
    try:
        genai_client.configure(api_key=API_KEY)
        models = genai_client.list_models()
        print("Models (from google.generativeai client):")
        for m in models:
            # models may be dict-like
            name = m.get('name') if isinstance(m, dict) else getattr(m, 'name', str(m))
            print(" -", name)
        sys.exit(0)
    except Exception as e:
        print("google.generativeai client present but failed to list models:", e)
except Exception:
    pass

# Fall back to REST API. The exact endpoint and parameters may vary depending on version.
# We'll try the v1beta endpoint which is commonly used with google genai.
print("Falling back to REST API call to list models...")

base_url = "https://generativelanguage.googleapis.com/v1beta/models"
# For list, some APIs support `?key=API_KEY` or parent parameter; we'll attempt a simple GET
url = f"{base_url}?key={urllib.parse.quote(API_KEY)}"

try:
    with urllib.request.urlopen(url) as resp:
        data = resp.read().decode('utf-8')
        parsed = json.loads(data)
        # parsed may contain 'models' or be a dict of models
        if isinstance(parsed, dict) and 'models' in parsed:
            print("Models (from REST):")
            for m in parsed['models']:
                print(" -", m.get('name'))
        else:
            # print raw for inspection
            print(json.dumps(parsed, indent=2))
except urllib.error.HTTPError as he:
    body = he.read().decode('utf-8')
    print(f"HTTP Error: {he.code} {he.reason}")
    print(body)
except Exception as e:
    print("Failed to list models with REST fallback:", e)

print("Done. If you see model names, set GEMINI_MODEL to one of them and re-run the Streamlit app.")
