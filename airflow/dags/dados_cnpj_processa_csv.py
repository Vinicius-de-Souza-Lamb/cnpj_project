"""
cnpj_ingest_dag.py
==================
End-to-end ingestion of the Receita Federal “Dados Abertos CNPJ” dump:

  1. Unzip raw archives → verbatim CSVs          (ZIP → CSV)
  2. (Re)create typed temp tables in Postgres    (DDL)
  3. COPY CSVs into their matching tables        (CSV → PG)
  4. Promote temp tables to final names          (temp_* → final)
  5. Harden final schemas (TEXT-ish dates → DATE)

Author: Vinicius Lamb
"""

# ═════════════════════════════════════════════════════════════════════
# 0)  Imports & global configuration
# ═════════════════════════════════════════════════════════════════════

from __future__ import annotations

import shutil
import logging
import os
import sys
import time
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from dotenv import load_dotenv
from postgre_connector import Postgres               # ← thin wrapper

load_dotenv()                                        # .env → os.environ

# ---- project folders --------------------------------------------------------
REPO_ROOT       = Path(__file__).resolve().parents[2]           # …/airflow/…
BASE_DIR        = REPO_ROOT / "airflow" / "datasets"
SOURCE_ZIP_DIR  = BASE_DIR / "public-zips"                      # produced by *download* DAG
CSV_DIR         = BASE_DIR / "public-data"                      # raw CSVs land here

for p in (SOURCE_ZIP_DIR, CSV_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---- logging ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s ➜ %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("cnpj-ingest-dag")
log.info("REPO_ROOT      ➜ %s", REPO_ROOT)
log.info("SOURCE_ZIP_DIR ➜ %s", SOURCE_ZIP_DIR)
log.info("CSV_DIR        ➜ %s", CSV_DIR)

# ──────────────────────────────────────────────────────────────────────
# 1)  Column layouts (Portuguese names)
# ──────────────────────────────────────────────────────────────────────
LAYOUTS: Dict[str, List[str]] = {
    "header": [
        "tipo_do_registro", "filler", "nome_do_arquivo", "data_de_gravacao",
        "numero_da_remessa", "filler2", "fim_de_registro"
    ],
    "cnae": ["codigo", "descricao"],
    "empresas": [
        "cnpj_basico", "razao_social", "natureza_juridica",
        "qualificacao_responsavel", "capital_social_str", "porte_empresa",
        "ente_federativo_responsavel"
    ],
    "estabelecimento": [
        "cnpj_basico", "cnpj_ordem", "cnpj_dv", "matriz_filial", 
        "nome_fantasia", "situacao_cadastral", "data_situacao_cadastral",
        "motivo_situacao_cadastral", "nome_cidade_exterior", "pais",
        "data_inicio_atividades", "cnae_fiscal", "cnae_fiscal_secundaria",
        "tipo_logradouro", "logradouro", "numero", "complemento", 
        "bairro", "cep", "uf", "municipio", "ddd1", "telefone1", 
        "ddd2", "telefone2", "ddd_fax", "fax", "correio_eletronico", 
        "situacao_especial", "data_situacao_especial"
    ],
    "motivo": ["codigo", "descricao"],
    "municipio": ["codigo", "descricao"],
    "natureza_juridica": ["codigo", "descricao"],
    "pais": ["codigo", "descricao"],
    "qualificacao_socio": ["codigo", "descricao"],
    "simples": [
        "cnpj_basico", "opcao_simples", "data_opcao_simples",
        "data_exclusao_simples", "opcao_mei", "data_opcao_mei",
        "data_exclusao_mei"
    ],
    "socios_original": [
        "cnpj_basico", "identificador_de_socio", "nome_socio", 
        "cnpj_cpf_socio", "qualificacao_socio", "data_entrada_sociedade", 
        "pais", "representante_legal", "nome_representante", 
        "qualificacao_representante_legal", "faixa_etaria"
    ],
    "cnaes_secundarias": [
        "tipo_do_registro", "indicador_full_diario", "tipo_de_atualizacao",
        "cnpj", "cnae_secundaria", "filler", "fim_de_registro"
    ],
    "trailler": [
        "tipo_do_registro", "filler", "total_de_registros_t1", "total_de_registros_t2",
        "total_de_registros_t3", "total_de_registros", "filler2", "fim_de_registro"
    ]
}

# ═════════════════════════════════════════════════════════════════════
# 2)  ZIP → CSV  (pure extraction, no transformation)
# ═════════════════════════════════════════════════════════════════════
def extract_zip_to_csv(limit: Optional[int] = None) -> None:
    """
    Decompress every ``*.zip`` found under :pydata:`SOURCE_ZIP_DIR`
    and write the *verbatim* contents to :pydata:`CSV_DIR`.

    One archive may contain multiple files – each gets an output name
    like:

        {zip_stem}_{inner_filename}.csv

    Examples
    --------
    >>> extract_zip_to_csv()        # process everything
    >>> extract_zip_to_csv(limit=2) # process only the first 2 ZIPs

    Parameters
    ----------
    limit : int | None, optional
        Maximum number of ZIP archives to process.  Passing *None*
        (default) means “process them all”.

    Notes
    -----
    * **No** data transformation happens here – we copy the bytes 1-for-1.
    * Already-extracted files are skipped (idempotent behaviour).
    * Detailed logging shows progress and per-file timing.
    """

    # ----------------------------------------------------------------
    # 0)  Gather and optionally truncate the list of ZIP archives
    # ----------------------------------------------------------------
    zip_paths = sorted(
        p for p in SOURCE_ZIP_DIR.iterdir()
        if p.suffix.lower() == ".zip"
    )

    if not zip_paths:
        log.warning("⚠️  No ZIP archives found in %s – nothing to extract.", SOURCE_ZIP_DIR)
        return

    if limit is not None:
        zip_paths = zip_paths[:limit]                    # truncate for quick tests

    total_archives = len(zip_paths)
    processed      = 0
    t_global_start = time.perf_counter()

    log.info("🔧  ZIP ➜ CSV extraction started (archives=%s)", total_archives)

    # ----------------------------------------------------------------
    # 1)  Walk each .zip and dump everything inside
    # ----------------------------------------------------------------
    for idx, zip_path in enumerate(zip_paths, start=1):
        t_start = time.perf_counter()
        log.info("→ [%s/%s]  %s", idx, total_archives, zip_path.name)

        try:
            with zipfile.ZipFile(zip_path) as zf:
                for inner_name in zf.namelist():
                    # Build the output filename:  {zip_stem}_{inner_basename}.csv
                    out_path = CSV_DIR / f"{zip_path.stem}_{Path(inner_name).name}.csv"

                    # Skip if we already have it – idempotency is ❤️, :)
                    if out_path.exists():
                        log.debug("   • skip (exists)  %s", out_path.name)
                        continue

                    # Copy the bytes verbatim (no decoding here)
                    with zf.open(inner_name) as src, out_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)  # 1 MiB chunks
                        

                    log.debug("   • extracted        %s", out_path.name)

            processed += 1
            log.info("   ✔  done in %.2fs", time.perf_counter() - t_start)

        except zipfile.BadZipFile as err:
            # Corrupted archive – log and carry on
            log.warning("⚠️  Ignored corrupt ZIP %s – %s", zip_path.name, err)
            continue

    # ----------------------------------------------------------------
    # 2)  Global summary
    # ----------------------------------------------------------------
    total_time = time.perf_counter() - t_global_start
    log.info("🏁  Extraction finished: %s / %s archive(s) processed in %.2fs",
             processed, total_archives, total_time)


# ═════════════════════════════════════════════════════════════════════
# 3)  Layout resolver · filename → LAYOUTS key (case-insensitive)
# ═════════════════════════════════════════════════════════════════════
def layout_key(file_name: str) -> Optional[str]:
    """
    Inspect *file_name* and return the matching entry inside
    the global :pydata:`LAYOUTS` dict.

    The match is intentionally **simple**:  
    we look for each layout keyword (``"empresas"``, ``"cnae"``,
    ``"estabelecimento"`` …) as a *substring* of the file name
    – case-insensitive.  The **first** hit wins.

    Parameters
    ----------
    file_name : str
        Any file name or full path (only the final component is used).

    Returns
    -------
    str | None
        * The layout key (e.g. ``"empresas"``) when a match is found;
        * **None** when no keyword matches.
    """
    # Normalize: keep just the final component and lower-case it
    fname_low = Path(file_name).name.lower()

    # 1) Direct match against our canonical keys in LAYOUTS
    #    (layout keys already reflect business names we use in tables)
    for key in LAYOUTS:                  # LAYOUTS is defined at module top
        if key in fname_low:
            return key                   # ← success

    # 2) Common aliases found in Receita filenames (abbreviations)
    #    We keep this list local to the function for clarity/encapsulation.
    #    Order matters only if two aliases could match the same file name.
    aliases = {
        # canonical_key              # examples of filename substrings
        "natureza_juridica": ["natju", "natureza"],
        "qualificacao_socio": ["quals", "qualific"],
        "socios_original":    ["socio", "socios"],
        "estabelecimento":    ["estabelec", "estab"],
        "empresas":           ["empre", "empresa", "empresas"],
        "cnae":               ["cnae"],
        "motivo":             ["motic", "motivo"],
        "municipio":          ["munic", "municipio"],
        "pais":               ["pais"],
        "cnaes_secundarias":  ["secundaria", "secundarias"],
        "trailler":           ["trailler", "trailer"],
        "header":             ["header", "cabecalho"],
    }

    for canonical, subs in aliases.items():
        if any(sub in fname_low for sub in subs):
            return canonical

    # Fallback – unknown file
    return None


# ═════════════════════════════════════════════════════════════════════
# 4)  DDL · (Re)create temp_* tables (loose typing for ingestion)
# ═════════════════════════════════════════════════════════════════════
def create_temp_tables() -> None:
    """
    Build/rebuild every **temp_*** table in PostgreSQL using the
    *true* data-types published by Receita Federal.  
    All DDLs start with ``DROP TABLE IF EXISTS`` ➜ the helper is
    **idempotent** and can run safely on every DAG execution.
    """
    log.info("🔧  (Re)creating typed temp tables …")

    ddl_statements = [
        # ------------------------------------------------------------------
        # cnae / industry_classification
        # ------------------------------------------------------------------
        """
        DROP TABLE IF EXISTS temp_cnae CASCADE;
        CREATE TABLE temp_cnae (
            codigo     CHAR(7),
            descricao  TEXT     NOT NULL
        );
        """,
        # ------------------------------------------------------------------
        # empresas / companies
        # ------------------------------------------------------------------
        """
        DROP TABLE IF EXISTS temp_empresas CASCADE;
        CREATE TABLE temp_empresas (
            cnpj_basico                 CHAR(8),
            razao_social                TEXT       NOT NULL,
            natureza_juridica           CHAR(4)    NOT NULL,
            qualificacao_responsavel    CHAR(2)    NOT NULL,
            capital_social              VARCHAR(20),
            porte_empresa               CHAR(2),
            ente_federativo_responsavel TEXT
        );
        """,

        # ------------------------------------------------------------------
        # estabelecimento / establishments
        # ------------------------------------------------------------------
"""
        DROP TABLE IF EXISTS temp_estabelecimento CASCADE;
        CREATE TABLE temp_estabelecimento (
            cnpj_basico               CHAR(8),
            cnpj_ordem                CHAR(4),
            cnpj_dv                   CHAR(2),
            matriz_filial             CHAR(1),
            nome_fantasia             TEXT,
            situacao_cadastral        CHAR(2),
            data_situacao_cadastral   TEXT,
            motivo_situacao_cadastral CHAR(2),
            nome_cidade_exterior      TEXT,
            pais                      CHAR(3),
            data_inicio_atividades    TEXT,
            cnae_fiscal               CHAR(7),
            cnae_fiscal_secundaria    TEXT,
            tipo_logradouro           TEXT,
            logradouro                TEXT,
            numero                    VARCHAR(10),
            complemento               TEXT,
            bairro                    TEXT,
            cep                       CHAR(8),
            uf                        CHAR(2),
            municipio                 CHAR(4),
            ddd1                      CHAR(4),
            telefone1                 CHAR(8),
            ddd2                      CHAR(4),
            telefone2                 CHAR(8),
            ddd_fax                   CHAR(4),
            fax                       CHAR(8),
            correio_eletronico        TEXT,
            situacao_especial         TEXT,
            data_situacao_especial    TEXT
        );
        """,

        # ------------------------------------------------------------------
        # motivo / status_reason
        # ------------------------------------------------------------------
        """
        DROP TABLE IF EXISTS temp_motivo CASCADE;
        CREATE TABLE temp_motivo (
            codigo    CHAR(2),
            descricao TEXT    NOT NULL
        );
        """,

        # ------------------------------------------------------------------
        # municipio / municipalities
        # ------------------------------------------------------------------
        """
        DROP TABLE IF EXISTS temp_municipio CASCADE;
        CREATE TABLE temp_municipio (
            codigo    CHAR(4),
            descricao TEXT NOT NULL
        );
        """,

        # ------------------------------------------------------------------
        # natureza_juridica / legal_nature
        # ------------------------------------------------------------------
        """
        DROP TABLE IF EXISTS temp_natureza_juridica CASCADE;
        CREATE TABLE temp_natureza_juridica (
            codigo    CHAR(4),
            descricao TEXT NOT NULL
        );
        """,

        # ------------------------------------------------------------------
        # pais / countries
        # ------------------------------------------------------------------
        """
        DROP TABLE IF EXISTS temp_pais CASCADE;
        CREATE TABLE temp_pais (
            codigo    CHAR(3),
            descricao TEXT NOT NULL
        );
        """,

        # ------------------------------------------------------------------
        # qualificacao_socio / partner_qualification
        # ------------------------------------------------------------------
        """
        DROP TABLE IF EXISTS temp_qualificacao_socio CASCADE;
        CREATE TABLE temp_qualificacao_socio (
            codigo    CHAR(2),
            descricao TEXT NOT NULL
        );
        """,

        # ------------------------------------------------------------------
        # simples / simple_tax_regime
        # ------------------------------------------------------------------
        """
        DROP TABLE IF EXISTS temp_simples CASCADE;
        CREATE TABLE temp_simples (
            cnpj_basico            CHAR(8),
            opcao_simples          CHAR(1),
            data_opcao_simples     TEXT,
            data_exclusao_simples  TEXT,
            opcao_mei              CHAR(1),
            data_opcao_mei         TEXT,
            data_exclusao_mei      TEXT
        );
        """,

        # ------------------------------------------------------------------
        # socios_original / original_partners
        # ------------------------------------------------------------------
        """
        DROP TABLE IF EXISTS temp_socios_original CASCADE;
        CREATE TABLE temp_socios_original (
            cnpj_basico                      CHAR(8),
            identificador_de_socio           CHAR(1),
            nome_socio                       TEXT,
            cnpj_cpf_socio                   CHAR(14),
            qualificacao_socio               CHAR(2),
            data_entrada_sociedade           TEXT,
            pais                             CHAR(3),
            representante_legal              CHAR(11),
            nome_representante               TEXT,
            qualificacao_representante_legal CHAR(2),
            faixa_etaria                     CHAR(1)
        );
        """
    ]

    pg = Postgres()
    for ddl in ddl_statements:
        log.debug("Executing DDL:\n%s", ddl.strip())
        pg.execute_sql(ddl)

    log.info("✅  Temp tables successfully (re)created.")


# ════════════════════════════════════════════════════════════════════
# 5)  Low-level I/O · robust COPY with encoding detection (UTF-8/LATIN-1)
#                 and disk-backed NUL-strip fallback
# ════════════════════════════════════════════════════════════════════
def _has_nul_bytes(p: Path, probe_bytes: int = 64 * 1024) -> bool:
    """
    Quickly probe for rogue NUL bytes in a CSV file.

    Rationale
    ---------
    PostgreSQL COPY rejects NUL bytes. Scanning the entire file would be
    wasteful, so we heuristically check only the first *probe_bytes*.
    In practice, if NUL appears in the head, it tends to appear elsewhere.

    Parameters
    ----------
    p : Path
        CSV file path to probe.
    probe_bytes : int, default 64 KiB
        Number of bytes to read from the head of the file.

    Returns
    -------
    bool
        True  -> at least one NUL ('\\x00') found in the probed slice.
        False -> no NUL found in the slice (not a formal proof of absence).
    """
    with p.open("rb") as fh:
        return b"\x00" in fh.read(probe_bytes)

def _detect_csv_encoding(p: Path, probe_bytes: int = 128 * 1024) -> str:
    """
    Decide the best text encoding for a CSV file between **UTF-8** and **Latin-1**.

    Strategy
    --------
    • Read only the first *probe_bytes* to decide quickly (no full scan).
    • If there is a UTF-8 BOM (EF BB BF) → return **'utf-8-sig'**.
    • Else, try to decode that slice as UTF-8:
        – If it succeeds → return **'utf-8'**.
        – If it raises UnicodeDecodeError → fall back to **'latin-1'**.
      (Latin-1 never raises on arbitrary bytes, so order matters.)

    Parameters
    ----------
    p : Path
        Path to the CSV file on disk.
    probe_bytes : int, default 128 KiB
        Number of bytes to sample from the file head.

    Returns
    -------
    str
        One of: **'utf-8-sig'**, **'utf-8'**, or **'latin-1'**.
    """
    with p.open("rb") as fh:
        head = fh.read(probe_bytes)

    # BOM present → UTF-8 with signature
    if head.startswith(b"\xEF\xBB\xBF"):
        return "utf-8-sig"

    # Try a plain UTF-8 decode on the sample
    try:
        head.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"

def _copy_csv_to_pg(csv_path: Path, target_table: str) -> None:
    """
    Stream **one** CSV into PostgreSQL via `COPY … FROM STDIN` (CSV mode),
    with **per-file encoding detection** and a **NUL-strip** disk-backed fallback.

    Robustness & memory-safety
    --------------------------
    1) **Encoding detection per file** (UTF-8 or Latin-1) to avoid mojibake
       in the dimension tables (e.g., acentos).
    2) **Fast path**: stream the file with the detected encoding directly
       into `COPY` (low memory, no materialization).
    3) **NUL handling**: if COPY fails and the error/probe suggests NUL bytes,
       write a **temporary cleaned file** stripping only `\\x00` line-by-line,
       then COPY from it. Avoids building giant in-memory buffers.
    4) **Per-table CSV tweaks**: use `FORCE_NULL` on known date-like columns
       so that `""` becomes SQL NULL *before* casting happens downstream.

    Receita CNPJ CSV specifics
    --------------------------
    • Delimiter: `;`
    • No header row (HEADER false)
    • Empty strings `""` → NULL (NULL '')
    • Some files are UTF-8; others Latin-1 (ISO-8859-1)

    Parameters
    ----------
    csv_path : Path
        Absolute path to the CSV file to ingest.
    target_table : str
        Destination temp table name (e.g., `'temp_empresas'`).

    Raises
    ------
    Exception
        Any non-NUL related COPY error; or a failure that persists even
        after the NUL-strip fallback.
    """
    # ── Per-table CSV options kept local (encapsulation) ────────────────────
    _COPY_PER_TABLE_OPTS = {
        "temp_estabelecimento": (
            "FORCE_NULL (data_situacao_cadastral, data_inicio_atividades, data_situacao_especial)"
        ),
        "temp_simples": (
            "FORCE_NULL (data_opcao_simples, data_exclusao_simples, data_opcao_mei, data_exclusao_mei)"
        ),
        "temp_socios_original": "FORCE_NULL (data_entrada_sociedade)",
        # add others if new date-like TEXT columns appear
    }

    # 0) Detect per-file encoding and map to Postgres client_encoding
    file_encoding = _detect_csv_encoding(csv_path)
    client_enc   = "UTF8" if file_encoding.startswith("utf-") else "LATIN1"

    pg   = Postgres()                     # thin wrapper (env via dotenv)
    conn = pg.connect_postgres()
    cur  = conn.cursor()

    # Session knobs for predictable COPY STDIN behavior
    cur.execute(f"SET client_encoding TO '{client_enc}';")   # COPY STDIN uses session encoding
    cur.execute("SET datestyle TO ISO, YMD;")                # predictable date parsing downstream

    base_opts  = "FORMAT csv, DELIMITER ';', NULL '', HEADER false"
    extra_opts = _COPY_PER_TABLE_OPTS.get(target_table, "")
    copy_sql   = f"COPY {target_table} FROM STDIN WITH ({base_opts}{', ' + extra_opts if extra_opts else ''})"

    try:
        # ── 1) Fast path — raw stream in detected text encoding ────────────
        with csv_path.open("r", encoding=file_encoding, errors="replace", newline="") as fh:
            cur.copy_expert(copy_sql, fh)
        conn.commit()
        log.debug("✓ raw COPY (%s)  %s → %s", file_encoding, csv_path.name, target_table)
        return

    except Exception as exc_raw:
        # ── 2) Decide whether we should attempt the NUL-cleaning fallback ──
        conn.rollback()
        msg = str(exc_raw).lower()
        nul_suspect = ("nul" in msg) or ("invalid byte" in msg) or _has_nul_bytes(csv_path)

        if not nul_suspect:
            # Not a NUL/low-level byte issue → surface the real problem
            log.error("❌ COPY failed for %s (non-NUL error) – %s", csv_path.name, exc_raw)
            raise

        log.warning("COPY failed for %s (%s). Retrying with NUL-stripped temp file…",
                    csv_path.name, exc_raw)

        # ── 3) Disk-backed cleaning pass (constant memory) ─────────────────
        import tempfile
        with tempfile.NamedTemporaryFile("w+", encoding=file_encoding, delete=False) as tmp:
            tmp_path = Path(tmp.name)

            # Stream read → strip only NULs → stream write
            with csv_path.open("r", encoding=file_encoding, errors="replace", newline="") as src:
                for line in src:
                    tmp.write(line.replace("\x00", ""))

            tmp.flush()
            tmp.seek(0)

            try:
                cur.copy_expert(copy_sql, tmp)
                conn.commit()
                log.debug("✓ cleaned COPY (%s)  %s → %s", file_encoding, csv_path.name, target_table)
            except Exception as exc_clean:
                conn.rollback()
                log.error("❌ COPY still failing for %s after NUL-strip – %s",
                          csv_path.name, exc_clean)
                raise
            finally:
                # Best-effort cleanup
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    log.warning("⚠️ Could not delete temp file %s", tmp_path)

    finally:
        # ── 4) Always close resources ───────────────────────────────────────
        try:
            cur.close()
        finally:
            conn.close()
        log.debug("Closed cursor/connection for %s", csv_path.name)

# ════════════════════════════════════════════════════════════════════
# 6)  High-level loader · iterate CSV_DIR and COPY into matching temp tables
# ════════════════════════════════════════════════════════════════════
def load_csvs_into_postgres(limit_files: Optional[int] = None) -> bool:
    """
    COPY every ``*.csv`` sitting in :pydata:`CSV_DIR` into its matching
    *temp_* table (mapping is resolved via :func:`layout_key`).

    Parameters
    ----------
    limit_files : int | None, optional
        * **None** (default) – ingest **all** CSVs present.  
        * **N ≥ 1**        – stop after ingesting N files (super handy
          for unit tests, CI pipelines, or “quick-and-dirty” notebook
          sessions).

    Returns
    -------
    bool
        **True**  – every file loaded successfully.  
        **False** – at least one load failed (check the logs).

    Implementation details
    ----------------------
    • Files are processed in **alphabetical order** for deterministic
      behaviour (important in CI).  
    • Each file is COPY-ed through the low-level helper
      :func:`_copy_csv_to_pg`, which already handles NUL-byte issues.  
    • The function is *pure* – it only logs and returns a boolean; the
      surrounding Airflow task decides what to do with the status.
    """
    # 1)  Gather CSV paths ----------------------------------------------------
    csv_paths = sorted(p for p in CSV_DIR.iterdir() if p.suffix.lower() == ".csv")
    if limit_files is not None:
        csv_paths = csv_paths[:limit_files]

    if not csv_paths:
        log.error("⚠️  No CSV files found in %s – nothing to load.", CSV_DIR)
        return False

    # 2)  Loop & COPY each file ---------------------------------------------
    total       = len(csv_paths)
    processed   = 0
    all_success = True

    for idx, csv_path in enumerate(csv_paths, start=1):
        key = layout_key(csv_path.name)
        if not key:
            log.error("❌  Unknown layout for %s – skipped.", csv_path.name)
            all_success = False
            continue

        dest_table = f"temp_{key}"
        log.info("→ [%s/%s] %s  →  %s", idx, total, csv_path.name, dest_table)

        try:
            _copy_csv_to_pg(csv_path, dest_table)
            processed += 1
        except Exception as err:
            # The helper logs details; we just flag the failure here.
            log.error("❌  COPY failed for %s – %s", csv_path.name, err)
            all_success = False

    # 3)  Summary -------------------------------------------------------------
    log.info(
        "🏁  CSV loader finished – %s / %s file(s) processed – status: %s",
        processed,
        total,
        "SUCCESS" if all_success else "PARTIAL FAILURE",
    )
    return all_success


# ════════════════════════════════════════════════════════════════════
# 7)  Promotion · temp_* → final names (idempotent & transactional)
# ════════════════════════════════════════════════════════════════════
def promote_temp_tables(**ctx) -> None:
    """
    Swap every *temp_* table into its permanent counterpart – but **only
    if** the preceding CSV-loader task signalled **success** through
    XCom.

    Safety rails
    ------------
    • **Status gate:** promotion is **skipped** when the `load_csvs`
      task pushed a falsy `status` flag (meaning one or more COPY
      operations failed).  
    • **Transactional semantics:** each table swap happens inside its
      own implicit PostgreSQL transaction (one `execute_sql` call ⇒ one
      `commit`). If a single table fails, the remaining ones are still
      attempted and the error is clearly logged.  
    • **Idempotent by design:** we `DROP TABLE IF EXISTS …` first, so
      re-running the DAG for the same execution date will simply
      overwrite yesterday’s snapshot.

    Parameters
    ----------
    **ctx
        Airflow task-context dictionary – we only need it to pull the
        XCom key ``status`` emitted by *load_csvs*.
    """
    # ── 1)  Did the loader succeed? ──────────────────────────────────
    status: bool = ctx["ti"].xcom_pull(task_ids="load_csvs", key="status")
    if not status:
        log.error(
            "🚫  CSV loader reported errors – temp-table promotion ABORTED."
        )
        return

    # ── 2)  Promote each table individually ─────────────────────────
    pg = Postgres()
    promoted, errors = 0, 0

    for name in LAYOUTS.keys():
        temp_name = f"temp_{name}"
        log.info("🔄  Promoting  %s  →  %s", temp_name, name)

        try:
            pg.execute_sql(f"DROP TABLE IF EXISTS {name};")
            pg.execute_sql(f"ALTER TABLE {temp_name} RENAME TO {name};")
            promoted += 1
            log.debug("✓  %s promoted successfully", name)

        except Exception as exc:
            errors += 1
            log.error("❌  Failed promoting %s – %s", name, exc)

    # ── 3)  Final summary ───────────────────────────────────────────
    if errors == 0:
        log.info("🎉  Promotion complete – %s table(s) swapped.", promoted)
    else:
        log.warning(
            "⚠️  Promotion finished with %s error(s) – "
            "%s table(s) swapped, %s failed.",
            errors, promoted, errors
        )

# ════════════════════════════════════════════════════════════════════
# 7.1) Schema hardening · TEXT-ish dates → DATE with cleanup (safe/optional)
# ════════════════════════════════════════════════════════════════════
def _harden_final_schemas() -> None:
    """
    Convert TEXT-like date columns to proper DATE in final tables, cleaning common garbage:
    "", "0", "00000000", non-digits. Safe & idempotent:
      - Skips tables missing.
      - Skips columns already DATE.
      - For text-ish columns, casts using a robust CASE over col::text.
    """
    pg = Postgres()
    conn = pg.connect_postgres()
    cur  = conn.cursor()

    targets = {
        "estabelecimento": [
            "data_situacao_cadastral",
            "data_inicio_atividades",
            "data_situacao_especial",
        ],
        "simples": [
            "data_opcao_simples",
            "data_exclusao_simples",
            "data_opcao_mei",
            "data_exclusao_mei",
        ],
        "socios_original": [
            "data_entrada_sociedade",
        ],
    }

    def table_exists(tbl: str) -> bool:
        cur.execute("SELECT to_regclass('public.' || %s) IS NOT NULL;", (tbl,))
        return bool(cur.fetchone()[0])

    def column_type(tbl: str, col: str) -> str | None:
        cur.execute(
            """
            SELECT data_type
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s AND column_name=%s
            """,
            (tbl, col),
        )
        row = cur.fetchone()
        return row[0] if row else None  # e.g. 'text', 'character varying', 'date', ...

    # Expression that normalizes a TEXT-ish column into DATE.
    # Note: we always operate on col::text to support current type TEXT or DATE.
    def cast_expr(col: str) -> str:
        return f"""
        CASE
          WHEN NULLIF(regexp_replace(({col})::text, '\\D', '', 'g'), '') IS NULL
               OR NULLIF(regexp_replace(({col})::text, '\\D', '', 'g'), '') = '0'
            THEN NULL
          WHEN length(regexp_replace(({col})::text, '\\D', '', 'g')) = 8
            THEN to_date(regexp_replace(({col})::text, '\\D', '', 'g'), 'YYYYMMDD')
          WHEN ({col})::text ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}$'
            THEN (({col})::text)::date
          ELSE NULL
        END
        """

    altered_any = False

    for tbl, cols in targets.items():
        if not table_exists(tbl):
            log.info("↪️  Skip harden: table %s does not exist.", tbl)
            continue

        alters = []
        for col in cols:
            ctype = column_type(tbl, col)
            if ctype is None:
                log.info("↪️  Skip column: %s.%s not found.", tbl, col)
                continue
            if ctype.lower() == "date":
                log.debug("• %s.%s already DATE — skipping.", tbl, col)
                continue

            # Only non-DATE columns are altered
            alters.append(f"ALTER COLUMN {col} TYPE DATE USING {cast_expr(col)}")

        if alters:
            sql = f"ALTER TABLE {tbl} " + ", ".join(alters) + ";"
            cur.execute(sql)
            cur.execute(f"ANALYZE {tbl};")
            conn.commit()
            altered_any = True
            log.info("✅ Hardened %s (%d column(s)).", tbl, len(alters))
        else:
            log.info("↪️  Nothing to harden on %s.", tbl)

    if altered_any:
        log.info("🧱 Final schemas hardened (TEXT→DATE with cleanup).")
    else:
        log.info("ℹ️  No columns altered (already consistent).")

    try:
        cur.close()
    finally:
        conn.close()


# ════════════════════════════════════════════════════════════════════
# 8)  Airflow DAG definition
# ════════════════════════════════════════════════════════════════════
default_args = dict(
    owner="Vinicius Lamb",
    email_on_failure=True,
    start_date=days_ago(1),
    retries=0,
    retry_delay=timedelta(seconds=30),
)

with DAG(
    dag_id="dag_cnpj_ingest",
    schedule_interval="@monthly",
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    tags=["cnpj", "ingest"],
) as dag:

    unzip = PythonOperator(
        task_id="extract_zip_to_csv",
        python_callable=extract_zip_to_csv,
        op_kwargs={"limit": None},
    )

    ddl = PythonOperator(
        task_id="create_temp_tables",
        python_callable=create_temp_tables,
    )

    load = PythonOperator(
        task_id="load_csvs",
        python_callable=lambda **ctx: ctx["ti"].xcom_push(
            key="status",
            value=load_csvs_into_postgres(limit_files=None)
        ),
        retries=1,
    )

    promote = PythonOperator(
        task_id="promote_temp_tables",
        python_callable=promote_temp_tables,
    )

    harden = PythonOperator(
        task_id="harden_final_schemas",
        python_callable=_harden_final_schemas,
    )   

    unzip >> ddl >> load >> promote >> harden