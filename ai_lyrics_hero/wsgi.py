"""WSGI entry point for a production server such as Gunicorn."""

from .webapp import create_app

app = create_app()
