"""ollamazip — Bundle Ollama models into portable archives."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ollamazip")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
