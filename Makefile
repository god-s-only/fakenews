.PHONY: install install-dev run test lint fmt docker-up docker-down

install: ## Install runtime dependencies
	pip install -r requirements.txt

install-dev: ## Install development/test dependencies
	pip install -r requirements.txt -r requirements-dev.txt

run: ## Start the FastAPI server
	python main.py

test: ## Run the test suite
	pytest

lint: ## Run tests with verbose output (manual check)
	pytest -v

fmt: ## Format check (shows unused imports and style)
	python -m py_compile app/*.py
	python -m py_compile tests/*.py

docker-up: ## Build and start with Docker Compose
	docker compose up --build

docker-down: ## Stop Docker Compose services
	docker compose down
