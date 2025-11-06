# CNPJ Project — Airflow + Postgres + Streamlit

A reproducible data-engineering stack for Brazil’s **CNPJ** open data.  
It ships **Dockerized ETL pipelines** (Airflow), a **typed Postgres** warehouse, and a **modern Streamlit dashboard** (Plotly) — with a stylish **Geography** page (choropleth, KPIs, filters).

## Stack
- **Airflow 2.9** — orchestration of ingest/transform DAGs
- **Postgres 15** — durable warehouse (typed temp tables → promotion)
- **Streamlit** — dashboard (map, KPIs, CNAE, legal nature, etc.)
- **Docker Compose** — one-command local env

## Quick start
```bash
# 1) bring everything up
make init            # == docker compose up -d --build

# 2) open apps
open http://localhost:8080   # Airflow Web
open http://localhost:8501   # Streamlit

# logs
docker compose logs -f streamlit
docker compose logs -f airflow-web
