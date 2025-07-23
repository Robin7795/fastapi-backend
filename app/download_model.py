from sentence_transformers import SentenceTransformer

# 自動下載並儲存到指定目錄
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("./models/all-MiniLM-L6-v2")
