# 🇧🇷 CNPJ Data Pipeline — Airflow + PostgreSQL + Streamlit (Docker Compose)

A portfolio-grade **data engineering project** that ingests Brazil’s **Receita Federal CNPJ open data** (ZIP files), builds a **typed PostgreSQL warehouse** using a **staging → validation → promotion** pipeline, and serves an **analytics dashboard** in **Streamlit**.

This repository focuses on **real-world practices**: reproducible local environment, separation of concerns, read-only access for BI, transaction-safe promotion, and clear documentation.

---

## ✅ What this project delivers

- **End-to-end ETL**: download → unzip → staging → validate/transform → promote → consume
- **Typed staging schema (`temp_*`)** mirroring official Receita Federal layouts
- **Curated final tables** (clean, consistent, query-ready)
- **Streamlit analytics dashboard** querying **final tables** via **read-only DB user**
- **Docker Compose local environment** (fully reproducible)

---

## 🧱 Stack

- **Docker Compose** — local environment orchestration
- **Apache Airflow** — ETL orchestration and scheduling
  - Webserver (UI + API)
  - Scheduler (executes DAGs)
- **PostgreSQL 15** — warehouse
  - `airflow` database
  - staging + final schemas/tables
- **Streamlit** — analytics UI (KPIs, charts, filters, maps)

---

## 🧭 Architecture (Local)

![Local architecture](docs/diagrams/cnpj-local-architecture.png)

### Components and responsibilities

| Component | What it does | Why it matters |
|---|---|---|
| **External source (Receita Federal ZIPs)** | Public CNPJ dataset published as ZIP files | Real-world ingestion from external raw data |
| **Docker Compose network** | Runs everything together with consistent networking | Reproducible environment on any machine |
| **Airflow Webserver** | UI/API to monitor, trigger and debug DAGs | Visibility + operational control |
| **Airflow Scheduler** | Executes DAG tasks according to schedules | Orchestrates the pipeline reliably |
| **PostgreSQL 15** | Stores staging + final curated data | Durable analytics foundation |
| **Streamlit UI** | Dashboard consuming curated tables/views | Demonstrates data product delivery |

### Storage model

- **Postgres data** persists in a **Docker volume** (e.g., `pgdata`)
- **Airflow datasets/logs** use **bind mounts**
  - keeps logs accessible
  - supports log rotation
  - allows storing extracted files locally

---

## 🔐 Security & access model (real-world pattern)

This project enforces a safe access model:

### Database users

- **`airflow`** → **read/write**
  - used only by ETL tasks (load, transform, promote)
- **`app_ro`** → **read-only**
  - used only by Streamlit (analytics queries)

✅ Streamlit **never** has write permissions.  
✅ Staging is isolated from final tables.  
✅ Promotion is **transaction-safe** (rollback supported).

---

## 🔄 Data flow (ETL pipeline)

The pipeline always follows this sequence:

1. **Extract**
   - Download ZIP files via HTTP
   - Unzip raw CSVs
2. **Load**
   - Load raw data into **typed staging tables (`temp_*`)**
3. **Validate & Transform**
   - Fix invalid dates (`00000000`)
   - Normalize nulls
   - Enforce data types
   - Prepare referential integrity
4. **Promote**
   - Move validated data from `temp_*` into **final curated tables**
   - Use transactions (rollback-safe)
   - Keep pipeline **idempotent**
5. **Consume**
   - Streamlit queries final tables via **read-only user**
   - KPIs, charts, filters, maps

### Mermaid overview

```mermaid
erDiagram
  temp_empresas {
    CHAR(8) cnpj_basico PK
    CHAR(4) natureza_juridica FK
  }

  temp_estabelecimento {
    CHAR(8) cnpj_basico FK
    CHAR(7) cnae_fiscal FK
    CHAR(2) motivo_situacao_cadastral FK
    CHAR(4) municipio FK
    CHAR(3) pais FK
  }

  temp_socios_original {
    CHAR(8) cnpj_basico FK
    CHAR(2) qualificacao_socio FK
    CHAR(3) pais FK
  }

  temp_simples {
    CHAR(8) cnpj_basico FK
  }

  temp_cnae {
    CHAR(7) codigo PK
  }

  temp_motivo {
    CHAR(2) codigo PK
  }

  temp_municipio {
    CHAR(4) codigo PK
  }

  temp_natureza_juridica {
    CHAR(4) codigo PK
  }

  temp_pais {
    CHAR(3) codigo PK
  }

  temp_qualificacao_socio {
    CHAR(2) codigo PK
  }

  temp_empresas ||--o{ temp_estabelecimento : "cnpj_basico"
  temp_empresas ||--o{ temp_socios_original : "cnpj_basico"
  temp_empresas ||--o{ temp_simples : "cnpj_basico"

  temp_empresas }o--|| temp_natureza_juridica : "natureza_juridica"
  temp_estabelecimento }o--|| temp_cnae : "cnae_fiscal"
  temp_estabelecimento }o--|| temp_motivo : "motivo_situacao_cadastral"
  temp_estabelecimento }o--|| temp_municipio : "municipio"
  temp_estabelecimento }o--|| temp_pais : "pais"

  temp_socios_original }o--|| temp_qualificacao_socio : "qualificacao_socio"
  temp_socios_original }o--|| temp_pais : "pais"
