.PHONY: test format lint clean

# Format all Python files
format:
	@echo "==> Formatting Python files with ruff..."
	uv run ruff format .

# Run linters
lint:
	@echo "==> Checking Python file formatting..."
	uv run ruff format --check .
	@echo "==> Running ruff linter on Python files..."
	uv run ruff check .
	@echo "==> Running ty type checker..."
	uv run ty check . --error-on-warning --exclude="**/*.ipynb"

# Run all tests (format, lint, then pytest)
test: lint
	@echo "==> Running pytest..."
	uv run pytest

# Clean Python cache files
clean:
	@echo "==> Cleaning Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
