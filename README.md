[tool.poetry]
name = "autoresearcher"
version = "0.1.0"
description = "AutoResearcher – medical RAG assistant"
authors = ["<Sanjay> <<sanjaydv@umd.edu>>"]
packages = [{ include = "autoresearcher", from = "src" }]

[tool.poetry.dependencies]
python = "^3.10"
fastapi = "^0.110.0"
uvicorn = "^0.29.0"
pydantic = "^2.7.3"
pydantic-settings = "^2.2.1"
numpy = "^1.26.4"

[tool.poetry.group.dev.dependencies]
pytest = "^8.2.0"
ruff = "^0.4.4"
mypy = "^1.10.0"
types-requests = "^2.31.0.20240406"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
