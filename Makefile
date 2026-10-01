.PHONY: help dev infra gateway test eval lint clean stop

PYTHON := python
VENV := venv
ACTIVATE := source $(VENV)/bin/activate

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

infra: ## Start Redis + Postgres + Prometheus + Grafana via docker-compose
	docker compose up -d
	@echo ""
	@echo "Infrastructure started:"
	@echo "  Redis:        localhost:6379 (Insight UI at :8001)"
	@echo "  Postgres:     localhost:5432"
	@echo "  Prometheus:   http://localhost:9090"
	@echo "  Grafana:      http://localhost:3000"

gateway: ## Start the InferRoute gateway (port 8070)
	$(ACTIVATE) && uvicorn app.main:app --reload --port 8070

dev: ## Start everything — infra + gateway
	@echo "Run each in a separate terminal:"
	@echo "  Terminal 1: make infra"
	@echo "  Terminal 2: make gateway"
	@echo ""
	@echo "Then: curl http://localhost:8070/v1/health"

test: ## Run the full test suite (142 tests)
	$(ACTIVATE) && $(PYTHON) -m pytest tests/ -v

test-routing: ## Run only routing tests
	$(ACTIVATE) && $(PYTHON) -m pytest tests/routing/ -v

test-cache: ## Run only cache tests
	$(ACTIVATE) && $(PYTHON) -m pytest tests/cache/ -v

test-resilience: ## Run only resilience tests (circuit breaker, failover)
	$(ACTIVATE) && $(PYTHON) -m pytest tests/resilience/ -v

test-ratelimit: ## Run only rate limiting tests
	$(ACTIVATE) && $(PYTHON) -m pytest tests/ratelimit/ -v

test-rag: ## Run only RAG tests
	$(ACTIVATE) && $(PYTHON) -m pytest tests/rag/ -v

test-mcp: ## Run only MCP gateway tests
	$(ACTIVATE) && $(PYTHON) -m pytest tests/mcp/ -v

stop: ## Stop docker infrastructure
	docker compose down

clean: ## Stop and remove docker volumes
	docker compose down -v
