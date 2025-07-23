from sentence_transformers import SentenceTransformer
import os

# 📌 指定本地模型路徑
MODEL_PATH = os.path.join(os.getcwd(), "models", "all-MiniLM-L6-v2")

# ✅ 載入本地模型（需先手動下載）
try:
    model = SentenceTransformer(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"❌ 無法載入本地模型 '{MODEL_PATH}'，請確認模型已存在。\n錯誤原因：{str(e)}")

# ✨ 向量化單一句子
def encode_text(text: str):
    return model.encode(text.strip(), normalize_embeddings=True)

# ✨ 向量化多個句子（批次）
def encode_batch(texts: list[str]):
    texts = [t.strip() for t in texts]
    return model.encode(texts, normalize_embeddings=True)

# 📁 回傳指定領域的 FAISS index 儲存路徑
def get_index_path(domain: str):
    base_dir = os.path.join("uploads", "index", domain)
    os.makedirs(base_dir, exist_ok=True)
    return {
        "index": os.path.join(base_dir, "faiss.index"),
        "meta": os.path.join(base_dir, "meta.pkl")
    }
