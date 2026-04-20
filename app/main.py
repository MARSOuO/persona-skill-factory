from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import get_config

config = get_config()
app = FastAPI(title=config.app_name, version=config.app_version)
app.include_router(api_router)
