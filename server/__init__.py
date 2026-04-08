"""
Email Triage Server Package
"""

from .app import app

__all__ = ["app", "main"]


def main():
    """Entry point for the email-triage-server command."""
    import os
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
