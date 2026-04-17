import requests

url = "http://localhost:8000/query"

files = {
    "audio_file": open("test.wav", "rb")
}

res = requests.post(url, files=files)

print(res.json())