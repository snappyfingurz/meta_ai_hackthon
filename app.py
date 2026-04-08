"""App entrypoint for Hugging Face Spaces.

Starts the FastAPI environment server on the configured port.
"""
import os

import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
