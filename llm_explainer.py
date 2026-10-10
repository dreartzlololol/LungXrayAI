import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "medllama2"

prompt = """
You are a medical AI assistant.
Explain the possible clinical interpretation of a chest X-ray image
that has been classified as pneumonia by an AI model.
Focus on radiographic patterns and diagnostic limitations.
"""

payload = {
    "model": MODEL_NAME,
    "prompt": prompt,
    "stream": False
}

response = requests.post(OLLAMA_URL, json=payload)
result = response.json()

print(result["response"])
