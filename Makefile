# Makefile for RAG API Project

.PHONY: help migrate migrate-dry migrate-list migrate-rollback test-schema docker-up docker-down optimize-db analyze-db setup-complete setup-dev test-ci

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Database migrations
migrate: ## Run dash assistant database migrations
	python -m app.dash_assistant.migrate

migrate-dry: ## Show what migrations would be executed (dry run)
	python -m app.dash_assistant.migrate --dry-run

migrate-list: ## List all migrations and their status
	python -m app.dash_assistant.migrate --list

migrate-rollback: ## Rollback the last migration
	python -m app.dash_assistant.migrate --rollback

# Testing
test-schema: ## Run dash assistant schema tests
	pytest tests/dash_assistant/test_schema.py -v

test-all: ## Run all tests
	pytest tests/ -v

# Docker commands
docker-up: ## Start all services with docker-compose
	docker-compose up -d

docker-down: ## Stop all services
	docker-compose down

docker-db: ## Start only database service
	docker-compose -f db-compose.yaml up -d

docker-api: ## Start only API service
	docker-compose -f api-compose.yaml up -d

# Development
install-deps: ## Install Python dependencies
	pip install -r requirements.txt
	pip install -r test_requirements.txt

lint: ## Run linting
	ruff check app/ tests/
	mypy app/

format: ## Format code
	black app/ tests/
	ruff format app/ tests/

# Dash assistant specific
dash-init: migrate ## Initialize dash assistant (run migrations)

dash-test: test-schema ## Test dash assistant schema

dash-dev: docker-db migrate ## Setup dash assistant for development (start DB + run migrations)

# Ingestion commands
ingest-complete: ## Run complete ingestion pipeline (requires all data files)
	python -m app.dash_assistant.ingestion.index_jobs --complete \
		--dashboards-csv tests/fixtures/superset/dashboards.csv \
		--charts-csv tests/fixtures/superset/charts.csv \
		--md-dir tests/fixtures/superset/md \
		--enrichment-yaml tests/fixtures/superset/enrichment.yaml

ingest-dashboards: ## Load dashboards from CSV
	python -m app.dash_assistant.ingestion.index_jobs --dashboards-csv tests/fixtures/superset/dashboards.csv

ingest-charts: ## Load charts from CSV
	python -m app.dash_assistant.ingestion.index_jobs --charts-csv tests/fixtures/superset/charts.csv

ingest-markdown: ## Load markdown documentation
	python -m app.dash_assistant.ingestion.index_jobs --md-dir tests/fixtures/superset/md

ingest-enrichment: ## Apply enrichment configuration
	python -m app.dash_assistant.ingestion.index_jobs --enrichment-yaml tests/fixtures/superset/enrichment.yaml

test-ingestion: ## Run ingestion tests
	pytest tests/dash_assistant/test_ingestion.py -v

# Database optimization
optimize-db: ## Optimize database after mass loading (run ANALYZE)
	psql -h localhost -U postgres -d rag_api -c "SELECT optimize_after_mass_loading();"

analyze-db: ## Run ANALYZE on all dash assistant tables
	psql -h localhost -U postgres -d rag_api -c "SELECT analyze_dash_assistant_tables();"

# Complete setup workflow
setup-complete: docker-db migrate ingest-complete optimize-db ## Complete setup: DB + migrations + data + optimization

# Quick development setup
setup-dev: docker-db migrate ## Quick dev setup: DB + migrations only

# CI testing
test-ci: ## Test CI workflow locally (requires running PostgreSQL)
	./test_ci_locally.sh
