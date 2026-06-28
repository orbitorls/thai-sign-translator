"""Compatibility entrypoint for `uvicorn tsl.serving.api:app`."""
from tsl.api.app import app

__all__ = ["app"]
