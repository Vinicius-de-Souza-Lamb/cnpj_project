# -----------------------------------------------------------------------------
# Makefile — local dev helpers (Docker Compose)
# -----------------------------------------------------------------------------

.PHONY: help init stop reset logs-web

help:
	@echo ""
	@echo "CNPJ Data Pipeline — commands"
	@echo ""
	@echo "  make init      Start the stack (build + up -d)"
	@echo "  make stop      Stop containers (down)"
	@echo "  make reset     Stop + remove volumes (down -v)  ⚠️ deletes Postgres data"
	@echo "  make logs-web  Tail Airflow webserver logs"
	@echo ""

init:
	docker compose up -d --build

stop:
	docker compose down

reset:
	docker compose down -v

logs-web:
	docker compose logs -f airflow-web