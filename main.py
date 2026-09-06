"""Fake News Detector entry point.

Runs the FastAPI application with uvicorn. Configuration is read from
environment variables (see ``app/config.py`` and ``.env.example``).
"""

import uvicorn

from app.config import settings
from app.logging_config import configure_logging
from app.main import app

if __name__ == "__main__":
    configure_logging()
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
