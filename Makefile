# Define variables
DOCKER_COMPOSE_FILE=fastapi_deploy/docker-compose.yml

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
	@echo "prefect-serve-cloud and fastapi-up need to be run in separate terminals"
	@echo ""

.PHONY: prefect-serve-cloud
prefect-serve-cloud:  ## Serve Model Train and Load to GCS flows to Prefect Cloud
	python prefect_orchestration/serve_flows.py

.PHONY: fastapi-build
fastapi-build:  ## Build FastAPI docker image
	docker-compose -f $(DOCKER_COMPOSE_FILE) build

.PHONY: fastapi-up
fastapi-up:  ## Start FastAPI server
	docker-compose -f $(DOCKER_COMPOSE_FILE) up

