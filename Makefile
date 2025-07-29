.PHONY: dev test lint format build docker-run

dev:
	poetry run uvicorn autoresearcher.api.main:app --reload --port 8000

test:
	poetry run pytest -q

lint:
	poetry run ruff check src

format:
	poetry run ruff format src && poetry run black src

build:
	docker build -t autoresearcher:dev .

docker-run:
	docker run -p 8000:8080 autoresearcher:dev

index:
	poetry run python -m autoresearcher.scripts.build_index --pmc_dir data/pmc --cord_dir data/cord --out_path indexes/medical.faiss
# Make sure to run `chmod +x src/autoresearcher/scripts/build_index.py` to make the script executable
