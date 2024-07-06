# Define variables
DOCKER_COMPOSE_FILE=docker-compose.yaml

# Default target when `make` is run without arguments
.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help message
	@echo ""
	@echo "Usage: make [option]"
	@echo ""
	@echo "Options:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "prefect-serve-cloud and start-all are not run in detached mode"
	@echo ""

.PHONY: prefect-serve-cloud
prefect-serve-cloud:  ## Serve Model Train and Load to GCS flows to Prefect Cloud
	python prefect_orchestration/serve_flows.py

.PHONY: build-all
build-all:  ## Build image with PostgreSQL, pgAdmin, Grafana, Data upload to db, FastAPI
	docker-compose -f $(DOCKER_COMPOSE_FILE) build

.PHONY: start-all
start-all:  ## Start services
	docker-compose -f $(DOCKER_COMPOSE_FILE) up

