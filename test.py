# import requests
# import os

# API_KEY = os.getenv("LLM_API_KEY", "")
# MODEL_NAME = "nousresearch/hermes-3-llama-3.1-405b:free" 

# response = requests.post(
#     "https://openrouter.ai/api/v1/chat/completions",
#     headers={"Authorization": f"Bearer {API_KEY}"},
#     json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hello"}]}
# )

# print("Mã trạng thái:", response.status_code)
# print("Chi tiết lỗi từ OpenRouter:", response.text)

# embedding 
import os

import requests

API_KEY = os.getenv("LLM_API_KEY", "")
MODEL_NAME = "openai/text-embedding-3-small" 

# Đổi URL thành endpoint chuẩn /embeddings
url = "https://openrouter.ai/api/v1/embeddings"

# Cấu trúc JSON chuẩn cho model embedding: dùng 'input' thay vì 'messages'
payload = {
    "model": MODEL_NAME,
    "input": "hello"
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, json=payload)

print("Mã trạng thái:", response.status_code)
print("Chi tiết phản hồi:", response.json())