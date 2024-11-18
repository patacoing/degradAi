from fastapi import FastAPI

from app.routers.infer import router as infer_router

app = FastAPI()

app.include_router(infer_router, prefix="/infer")