import requests

response = requests.post("http://127.0.0.1:8000/", params={"prompt": "美国为什么要加息"})
french_text = response.json()

print(french_text)

