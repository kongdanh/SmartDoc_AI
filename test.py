import os
import requests
from dotenv import load_dotenv

# 1. BẮT BUỘC PHẢI CÓ: Nạp file .env vào hệ thống
load_dotenv()

# 2. Lấy API Key
API_KEY = os.getenv("LLM_API_KEY", "")

# 3. KIỂM TRA NHANH: In thử 10 ký tự đầu xem đã lấy được key chưa
if API_KEY == "":
    print("❌ LỖI: Không tìm thấy LLM_API_KEY. Hãy kiểm tra lại file .env!")
    exit()
else:
    print(f"✅ Đã nạp được API Key bắt đầu bằng: {API_KEY[:15]}...")

MODEL_NAME = "openai/text-embedding-3-small" 
url = "https://openrouter.ai/api/v1/embeddings"

payload = {
    "model": MODEL_NAME,
    "input": "hello"
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

print("Đang gửi request lên OpenRouter...")
response = requests.post(url, headers=headers, json=payload)

print("Mã trạng thái:", response.status_code)
print("Chi tiết phản hồi:", response.json())