import requests

API_KEY = "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"
MODEL_NAME = "arcee-ai/trinity-large-preview:free" 

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hello"}]}
)

print("Mã trạng thái:", response.status_code)
print("Chi tiết lỗi từ OpenRouter:", response.text)