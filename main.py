from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router

app = FastAPI()

# ✅ CORS 設定（讓 Next.js 可連線）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 上線後建議限制來源，例如 ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 掛載主要 API 路由
app.include_router(router)

# ✅ 根路由測試
@app.get("/")
def read_root():
    return {"message": "🚀 FastAPI is working!"}
