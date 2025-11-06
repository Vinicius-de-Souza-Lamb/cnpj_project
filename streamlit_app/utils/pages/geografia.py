from __future__ import annotations

# ─────────────────────────────── Standard library ──────────────────────────────
import json
from pathlib import Path
from typing import Optional, Iterable, Sequence, Dict, Tuple, List
from datetime import date, datetime, timedelta

# ─────────────────────────────── Third-party deps ──────────────────────────────
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text
from sqlalchemy.engine import Engine
from streamlit.errors import StreamlitAPIException

# ─────────────────────────────── Project UI utilities ──────────────────────────

try:
    from utils.ui import (
        THEME, inject_theme, fmt_num, fmt_money,
        kpi_card, stat_tile, block, hr, map_frame, date_filter_compact
    )
except ImportError:

    from ui import (
        THEME, inject_theme, fmt_num, fmt_money,
        kpi_card, stat_tile, block, hr, map_frame, date_filter_compact
    )

# ───────────────────────────────── Page configuration ──────────────────────────
PAGE_TITLE = "Geography | CNPJ"
PAGE_ICON = "🗺️"
PAGE_LAYOUT = "wide"
SHOW_REGION_GAP = False


def _configure_page() -> None:
    """
    Configure Streamlit page and inject the shared theme.

    Notes
    -----
    • `set_page_config` must be called exactly once per run. If this module
      is imported/reloaded more than once (e.g., during dev), Streamlit can
      raise a `StreamlitAPIException`. We guard against that to keep dev
      experience smooth.
    """
    try:
        st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=PAGE_LAYOUT)
    except StreamlitAPIException:
        # Already configured elsewhere in the same session – safe to ignore.
        pass

    # Apply the project-wide visual theme (colors, fonts, CSS helpers).
    inject_theme(THEME)


# Apply configuration immediately on import of this page.
_configure_page()


# ───────────────────────────────  DB / IO helpers  ─────────────────────────────
@st.cache_resource(show_spinner=False)
def _engine() -> Engine:
    """
    Return a shared SQLAlchemy Engine for the app.

    Resolution order
    ----------------
    1) utils.db._engine  (project-wide factory)
    2) db._engine        (local fallback)

    The engine is cached as a Streamlit *resource* so the connection
    pool is reused across reruns (faster and avoids connection storms).
    """
    try:
        from utils.db import _engine as factory  # type: ignore
    except Exception:
        from db import _engine as factory  # type: ignore
    return factory()

def _df_or_none(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return None if df is None or empty; otherwise the df itself."""
    if df is None:
        return None
    try:
        return None if df.empty else df
    except Exception:
        return None


def _load_df(sql: str) -> pd.DataFrame:
    """
    Execute SQL and return a pandas.DataFrame.

    Resolution order
    ----------------
    1) utils.db.load_df (preferred)
    2) db.load_df       (fallback)
    3) pandas.read_sql  (final fallback using this module's _engine)

    Results are cached for 5 minutes to keep the UI responsive.
    """
    @st.cache_data(ttl=300, show_spinner=False)
    def _run(q: str) -> pd.DataFrame:
        # Try the project loader first
        try:
            from utils.db import load_df as project_loader  # type: ignore
            return project_loader(q)
        except Exception:
            pass

        # Then try the local loader
        try:
            from db import load_df as local_loader  # type: ignore
            return local_loader(q)
        except Exception:
            pass

        # Final fallback: direct read via pandas + this engine
        return pd.read_sql(q, _engine())

    return _run(sql)

def _quote_ident(name: str) -> str:
    """
    Safely quote an identifier that may be:
      • "table"
      • schema.table

    Returns a string with double quotes applied to each part:
      → '"table"'  or  '"schema"."table"'

    Implementation detail
    ---------------------
    We build the string via concatenation (no backslashes in f-string
    expressions), preventing the classic `f-string expression part cannot
    include a backslash` syntax error.
    """
    part = (name or "").strip()
    if "." in part:
        sch, tbl = [p.strip().strip('"') for p in part.split(".", 1)]
        return '"' + sch + '"."' + tbl + '"'
    cleaned = part.strip().strip('"')
    return '"' + cleaned + '"'

def _db_ping() -> bool:
    """
    Lightweight health check: returns True if the database responds to `SELECT 1`.
    """
    try:
        with _engine().connect() as c:
            return bool(c.execute(text("SELECT 1")).scalar())
    except Exception:
        return False

@st.cache_data(ttl=300, show_spinner=False)
def _list_user_schemas(include_public: bool = True) -> list[str]:
    """
    Return the list of *user* schemas available in the database.

    Parameters
    ----------
    include_public : bool, default True
        Whether to include the 'public' schema in the result.

    Notes
    -----
    • Excludes PostgreSQL internal schemas (pg_catalog, information_schema, pg_toast*).
    • Results are cached for 5 minutes to avoid repeated catalog scans.
    • Returns a sorted, de-duplicated list of schema names.
    """
    eng = _engine()
    sql = """
        SELECT nspname
        FROM pg_namespace
        WHERE nspname NOT IN ('pg_catalog','information_schema')
          AND nspname NOT LIKE 'pg_toast%'
        ORDER BY 1;
    """
    try:
        with eng.connect() as c:
            rows = c.execute(text(sql)).fetchall()
    except Exception:
        # If introspection fails (permissions, network blip), degrade gracefully.
        return ["public"] if include_public else []

    # Normalize and de-duplicate
    schemas = sorted({(r[0] or "").strip() for r in rows if (r and r[0])})

    if not include_public:
        schemas = [s for s in schemas if s != "public"]

    return schemas

@st.cache_data(ttl=300, show_spinner=False)
def _first_existing_table(candidates: list[str]) -> Optional[str]:
    """
    Return the first NON-EMPTY table among the provided base names (e.g., ['empresas','temp_empresas']),
    scanning across all *user* schemas. The search respects the order of `candidates`.

    Returns
    -------
    str | None
        Fully-quoted identifier '"schema"."table"' (or '"table"' if found via search_path),
        or None when no non-empty match is found.

    Notes
    -----
    • A table is considered "existing" only if SELECT 1 ... LIMIT 1 returns at least one row.
    • We first enumerate existing tables from information_schema to avoid undefined-table errors.
    • Results are cached for 5 minutes.
    """
    eng = _engine()
    if not candidates:
        return None

    # 1) Normalize requested base table names (strip quotes/whitespace)
    bases: list[str] = []
    for base in candidates:
        if not base:
            continue
        bases.append(base.strip().strip('"'))

    if not bases:
        return None

    # 2) Pull all user schemas (optionally includes 'public' per helper's default)
    schemas = _list_user_schemas(include_public=True)

    # 3) Discover which of the (schema, base) combos actually exist using information_schema
    #    This avoids hammering the DB with failing SELECTs for non-existent tables.
    in_list = ", ".join("'" + b.replace("'", "''") + "'" for b in set(bases))
    info_sql = f"""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
          AND table_schema IN ({", ".join("'" + s.replace("'", "''") + "'" for s in schemas)})
          AND table_name IN ({in_list});
    """
    try:
        with eng.connect() as c:
            rows = c.execute(text(info_sql)).fetchall()
    except Exception:
        # If catalog introspection fails (permissions/driver issue),
        # gracefully fall back to direct probing in the next step.
        rows = []

    existing: set[tuple[str, str]] = {(r[0], r[1]) for r in rows}

    # 4) Build a prioritized list of fully-quoted identifiers to probe for "non-empty"
    #    Priority = order of `candidates` across all schemas, then bare name (search_path).
    probe_list: list[str] = []

    for base in bases:
        # Prefer explicitly qualified names that we know exist
        for sch in schemas:
            if (sch, base) in existing:
                probe_list.append(f'"{sch}"."{base}"')
        # Also try bare name (covers search_path/public usage)
        probe_list.append(f'"{base}"')

    # 5) Probe each candidate and return the first with at least one row
    for fq in probe_list:
        try:
            with eng.connect() as c:
                # LIMIT 1 is enough to assert non-empty without scanning
                has_row = c.execute(text(f"SELECT 1 FROM {fq} LIMIT 1")).fetchone() is not None
                if has_row:
                    return fq
        except Exception:
            # Undefined table / permission issues → just try the next candidate
            continue

    return None

@st.cache_data(ttl=300, show_spinner=False)
def df_top_naturezas(
    fq_table: str,
    ufs: Sequence[str] | None,
    start: str,
    end: str,
    top_n: int = 8
):
    """
    Top legal-nature categories (from empresas → natureza_juridica) within the given scope.

    - Prefers final tables over temp_ (resolved by _first_existing_table).
    - Normalizes nature codes (strip non-digits → take first 4 digits; if shorter, left-pad with zeros).
    - Scopes by the establishments in the selected UF(s) and date range before joining to empresas.

    Parameters
    ----------
    fq_table : str
        Fully-quoted establishments table (e.g., '"public"."estabelecimento"').
    ufs : Sequence[str] | None
        Optional list of UF codes. None/empty means “all UFs”.
    start, end : str
        Inclusive date range in 'YYYY-MM-DD'.
    top_n : int
        Max number of categories to return.

    Returns
    -------
    pandas.DataFrame | None
        Columns: natureza (str), qty (int). None when the dimension is unavailable or empty.
    """
    uf_filter = _in_clause_or_all(ufs)

    # Resolve dimension tables (final first, then temp_*). If either is missing, abort gracefully.
    emp_tbl = _first_existing_table(["empresas", "temp_empresas"])
    nj_tbl  = _first_existing_table(["natureza_juridica", "temp_natureza_juridica"])
    if not emp_tbl or not nj_tbl:
        return None

    sql = f"""
    WITH base AS (
      -- Establishments constrained only by UF (here) to allow date range enforcement in the next step
      SELECT {DATE_DT_EXPR} AS dt, e.cnpj_basico
      FROM {fq_table} e
      WHERE 1=1 {uf_filter}
    ),
    scoped AS (
      -- Distinct companies present in the date range (avoid double counting by multiple establishments)
      SELECT DISTINCT cnpj_basico
      FROM base
      WHERE dt BETWEEN '{start}' AND '{end}'
    ),
    emp AS (
      -- Bring nature codes from empresas, normalize to 4-digit string
      SELECT
        s.cnpj_basico,
        CASE
          WHEN length(regexp_replace(CAST(emp.natureza_juridica AS TEXT), '\\D', '', 'g')) >= 4
            THEN SUBSTRING(regexp_replace(CAST(emp.natureza_juridica AS TEXT), '\\D', '', 'g') FROM 1 FOR 4)
          ELSE LPAD(NULLIF(regexp_replace(CAST(emp.natureza_juridica AS TEXT), '\\D', '', 'g'), ''), 4, '0')
        END AS nat_code
      FROM scoped s
      LEFT JOIN {emp_tbl} emp USING (cnpj_basico)
    ),
    dim AS (
      -- Normalize the dictionary codes to the same 4-digit format
      SELECT
        CASE
          WHEN length(regexp_replace(CAST(nj.codigo AS TEXT), '\\D', '', 'g')) >= 4
            THEN SUBSTRING(regexp_replace(CAST(nj.codigo AS TEXT), '\\D', '', 'g') FROM 1 FOR 4)
          ELSE LPAD(NULLIF(regexp_replace(CAST(nj.codigo AS TEXT), '\\D', '', 'g'), ''), 4, '0')
        END AS codigo_norm,
        nj.descricao
      FROM {nj_tbl} nj
    )
    SELECT
      COALESCE(dim.descricao, 'Unknown') AS natureza,
      COUNT(*)::bigint AS qty
    FROM emp
    LEFT JOIN dim ON dim.codigo_norm = emp.nat_code
    GROUP BY 1
    ORDER BY qty DESC
    LIMIT {int(top_n)};
    """

    try:
        df = _load_df(sql)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        # If the dimension exists but query fails (permissions, type issues), fail soft so UI can fallback.
        return None

def _estab_candidates() -> list[str]:
    """
    Build an ordered list of establishments-table candidates across all user schemas.

    Priority rules (earlier = higher priority):
      1) For every user schema: <schema>.estabelecimento   (final table)
      2) For every user schema: <schema>.temp_estabelecimento
      3) Any other table in user schemas whose name matches '%estabele%'

    Notes
    -----
    - Returns unquoted identifiers like "public.estabelecimento".
      Callers should feed each item through `_quote_ident()` before using in SQL.
    - Deduplication preserves the first occurrence (highest priority).
    - If catalog lookup fails, returns only the synthesized priority list.
    """
    eng = _engine()

    # 1) List user-visible schemas
    try:
        schemas = _list_user_schemas()
    except Exception:
        # If we cannot list schemas, assume default search_path (usually "public")
        schemas = ["public"]

    # 2) Build priority list: final first, then temp_ for each schema
    priority: list[str] = []
    for sch in schemas:
        sch_clean = sch.strip().strip('"')
        priority.append(f"{sch_clean}.estabelecimento")
    for sch in schemas:
        sch_clean = sch.strip().strip('"')
        priority.append(f"{sch_clean}.temp_estabelecimento")

    # 3) Discover additional matches from information_schema
    discovered: list[str] = []
    sql = """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
          AND table_schema NOT IN ('pg_catalog','information_schema')
          AND LOWER(table_name) LIKE '%estabele%'
        ORDER BY table_schema, table_name;
    """
    try:
        with eng.connect() as c:
            rows = c.execute(text(sql)).fetchall()
            discovered = [f"{r.table_schema}.{r.table_name}" for r in rows]
    except Exception:
        # Fallback: no discovered additions if catalog query fails
        discovered = []

    # 4) Merge (priority first) and deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for name in priority + discovered:
        # Normalize key for dedup (case-insensitive on schema/table)
        key = name.lower().strip().strip('"')
        if key not in seen:
            seen.add(key)
            ordered.append(name)

    return ordered

@st.cache_data(ttl=300, show_spinner=False)
def detect_estab_table(*, debug: bool = False) -> Optional[str]:
    """
    Detect a readable, non-empty 'estabelecimento' table.

    Strategy
    --------
    1) Build a prioritized candidate list via `_estab_candidates()`
       (final tables first, then temp_, then any '%estabele%').
    2) For each candidate, quote it with `_quote_ident(name)` and probe:
         SELECT 1 FROM <fq> LIMIT 1
       If it returns a row, consider it valid and return the quoted name.
    3) If none are valid, return None. When debug=True, print a helpful trace.

    Returns
    -------
    str | None
        Fully-quoted identifier (e.g. `"public"."estabelecimento"`) or None.
    """
    eng = _engine()
    attempts: list[str] = []
    first_error: Optional[str] = None

    for name in _estab_candidates():
        fq = _quote_ident(name)
        attempts.append(fq)
        try:
            with eng.connect() as c:
                # Lightweight non-emptiness probe; also checks read permission.
                has_row = c.execute(text(f"SELECT 1 FROM {fq} LIMIT 1")).fetchone() is not None
            if has_row:
                if debug:
                    st.caption(f"Using establishments table: {fq}")
                return fq
        except Exception as ex:
            # Record the first error we hit to show in debug mode later
            if first_error is None:
                first_error = f"{type(ex).__name__}: {ex}"

    if debug:
        st.error("Could not find a non-empty establishments table.")
        if attempts:
            st.code("Tried (in order):\n" + "\n".join(attempts), language="text")
        if first_error:
            st.caption(f"First error encountered while probing: {first_error}")

    return None

# ───────────────────────────────  GeoJSON loader  ──────────────────────────────
def _geojson_path() -> Path:
    """
    Locate the Brazil states GeoJSON file on disk.

    Search order (first hit wins):
      1) <repo>/streamlit_app/assets/br_states.geojson  (two levels up from this file)
      2) <cwd>/streamlit_app/assets/br_states.geojson
      3) <cwd>/assets/br_states.geojson

    Returns
    -------
    Path
        Absolute path to the GeoJSON.

    Raises
    ------
    FileNotFoundError
        If the file cannot be found in any of the expected locations.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "assets" / "br_states.geojson",
        Path.cwd() / "streamlit_app" / "assets" / "br_states.geojson",
        Path.cwd() / "assets" / "br_states.geojson",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Build a helpful message showing exactly where we looked
    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not locate 'br_states.geojson'. Please place it under one of:\n"
        f"{searched}\n"
        "Tip: commit it to 'streamlit_app/assets/br_states.geojson' in your repo."
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_geojson() -> dict:
    """
    Load and lightly-validate the Brazil states GeoJSON.

    Returns
    -------
    dict
        Parsed GeoJSON object (FeatureCollection expected).

    Notes
    -----
    - Cached for 1 hour to avoid repeated disk I/O.
    - Performs minimal validation: expects a FeatureCollection with 'features'.
    """
    path = _geojson_path()
    with open(path, "r", encoding="utf-8") as fh:
        geo = json.load(fh)

    # Minimal validation to fail fast with a clear message
    if not isinstance(geo, dict) or geo.get("type") != "FeatureCollection" or "features" not in geo:
        raise ValueError(
            f"Invalid GeoJSON at '{path}': expected a FeatureCollection with a 'features' array."
        )

    if not geo["features"]:
        st.warning(f"GeoJSON loaded from '{path}', but it contains no features.")

    return geo


# ───────────────────────────────  Reference dictionaries  ─────────────────────────────
# Official IBGE macro-regions (by UF acronym).
# N  = North, NE = Northeast, CO = Center-West, SE = Southeast, S = South
UF_TO_REGION: Dict[str, str] = {
    # North
    "RO": "N", "AC": "N", "AM": "N", "RR": "N", "PA": "N", "AP": "N", "TO": "N",
    # Northeast
    "MA": "NE", "PI": "NE", "CE": "NE", "RN": "NE", "PB": "NE",
    "PE": "NE", "AL": "NE", "SE": "NE", "BA": "NE",
    # Center-West
    "MT": "CO", "MS": "CO", "GO": "CO", "DF": "CO",
    # Southeast
    "SP": "SE", "RJ": "SE", "ES": "SE", "MG": "SE",
    # South
    "PR": "S", "SC": "S", "RS": "S",
}

# Human-readable labels for those region codes (optional helper).
REGION_LABEL: Dict[str, str] = {
    "N": "North",
    "NE": "Northeast",
    "CO": "Center-West",
    "SE": "Southeast",
    "S": "South",
}

# Set of coastal UFs (touching the Atlantic).
COASTAL_UF: set[str] = {
    "AP", "PA", "MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE",
    "BA", "ES", "RJ", "SP", "PR", "SC", "RS",
}

def is_coastal(uf: str) -> bool:
    """
    Quick predicate: does this UF border the Atlantic?
    """
    return uf.upper() in COASTAL_UF


# ───────────────────────── Date coercion (TEXT/DATE safe for SQL) ─────────────────────
# This fragment normalizes `e.data_inicio_atividades` to a DATE inside SQL.
# It safely handles:
#   - TEXT with non-digits (we strip everything but digits and take first 8 as YYYYMMDD)
#   - ISO date string 'YYYY-MM-DD'
#   - Returns NULL when it cannot be parsed
#
# Usage in SQL:
#   SELECT
#     {DATE_DT_EXPR} AS dt,
#     ...
#   FROM some_table e
#
DATE_DT_EXPR = r"""
CASE
  WHEN regexp_replace(CAST(e.data_inicio_atividades AS TEXT), '\D', '', 'g') ~ '^[0-9]{8}'
    THEN to_date(SUBSTRING(regexp_replace(CAST(e.data_inicio_atividades AS TEXT), '\D', '', 'g'), 1, 8), 'YYYYMMDD')
  WHEN CAST(e.data_inicio_atividades AS TEXT) ~ '^\d{4}-\d{2}-\d{2}$'
    THEN (CAST(e.data_inicio_atividades AS TEXT))::date
  ELSE NULL
END
"""

# ─────────────────────────────────── Queries ───────────────────────────────────

def _in_clause_or_all(ufs: Sequence[str] | None) -> str:
    """
    Build a safe SQL fragment to filter by UF.
    - Returns an empty string when no filter is needed.
    - Normalizes inputs to UPPER(TRIM(value)).
    - Escapes single quotes to avoid breaking the SQL literal.
    NOTE: This fragment references the column as `e.uf` (the typical alias used in queries).
          If you need a different alias/column, adapt the function or pass through a formatter.
    """
    if not ufs:
        return ""

    # Normalize and deduplicate input UFs
    cleaned: list[str] = []
    seen = set()
    for u in ufs:
        if u is None:
            continue
        v = str(u).strip().upper().replace("'", "''")  # escape single quotes
        if v and v not in seen:
            seen.add(v)
            cleaned.append(v)

    if not cleaned:
        return ""

    # Compare using UPPER(TRIM(e.uf)) to be resilient to messy data
    literals = ",".join(f"'{v}'" for v in cleaned)
    return f" AND UPPER(TRIM(CAST(e.uf AS TEXT))) IN ({literals}) "


@st.cache_data(ttl=300, show_spinner=False)
def df_uf_stock(fq_table: str, start: str, end: str):
    """
    Aggregate the current stock of establishments by UF for the given date window.
    - Robust date parsing via DATE_DT_EXPR.
    - Normalizes UF to UPPER(TRIM(...)) to avoid duplicate keys like 'sp' vs 'SP '.
    - Excludes empty / NULL UF values.
    """
    sql = f"""
    WITH base AS (
      SELECT
        {DATE_DT_EXPR} AS dt,
        UPPER(TRIM(CAST(e.uf AS TEXT))) AS uf
      FROM {fq_table} e
    )
    SELECT uf, COUNT(*)::bigint AS qty
    FROM base
    WHERE uf IS NOT NULL AND uf <> '' AND dt BETWEEN '{start}' AND '{end}'
    GROUP BY uf
    ORDER BY qty DESC;
    """
    return _load_df(sql)

@st.cache_data(ttl=300, show_spinner=False)
def df_municipios_in_uf(
    fq_table: str,
    uf: str,
    start: str,
    end: str,
    limit_n: int = 50
):
    """
    Rank municipalities within a given UF for the selected date window.

    - Normalizes UF comparisons with UPPER(TRIM()).
    - Uses DATE_DT_EXPR to parse dates safely (TEXT or DATE).
    - Joins against the best-available municipality dimension table
      (“municipio” or “temp_municipio”). If neither exists, falls back
      to the raw code column and returns it as text.

    Returns a DataFrame with columns: ["municipio", "qty"] sorted desc.
    """
    # Sanitize/normalize inputs
    uf_lit = (uf or "").strip().upper().replace("'", "''")
    limit_n = max(1, int(limit_n))

    mun_tbl = _first_existing_table(["municipio", "temp_municipio"])

    if mun_tbl:
        # Robust join: compare trimmed TEXT to handle code stored as numeric/text
        sql = f"""
        WITH base AS (
          SELECT
            {DATE_DT_EXPR} AS dt,
            UPPER(TRIM(CAST(e.uf AS TEXT))) AS uf,
            TRIM(CAST(e.municipio AS TEXT)) AS mun_code
          FROM {fq_table} e
        ),
        inrng AS (
          SELECT * FROM base
          WHERE uf = '{uf_lit}' AND dt IS NOT NULL AND dt BETWEEN '{start}' AND '{end}'
        ),
        agg AS (
          SELECT
            m.descricao AS municipio,
            COUNT(*)::bigint AS qty
          FROM inrng b
          LEFT JOIN {mun_tbl} m
            ON TRIM(CAST(m.codigo AS TEXT)) = b.mun_code
          WHERE m.descricao IS NOT NULL AND m.descricao <> ''
          GROUP BY 1
        )
        SELECT municipio, qty
        FROM agg
        ORDER BY qty DESC
        LIMIT {limit_n};
        """
    else:
        # Fallback without dimension: show the raw municipality code as text
        sql = f"""
        WITH base AS (
          SELECT
            {DATE_DT_EXPR} AS dt,
            UPPER(TRIM(CAST(e.uf AS TEXT))) AS uf,
            TRIM(CAST(e.municipio AS TEXT)) AS municipio
          FROM {fq_table} e
        )
        SELECT municipio, COUNT(*)::bigint AS qty
        FROM base
        WHERE uf = '{uf_lit}'
          AND municipio IS NOT NULL AND municipio <> ''
          AND dt IS NOT NULL AND dt BETWEEN '{start}' AND '{end}'
        GROUP BY 1
        ORDER BY qty DESC
        LIMIT {limit_n};
        """

    return _load_df(sql)


@st.cache_data(ttl=300, show_spinner=False)
def df_openings_by_year(
    fq_table: str,
    ufs: Sequence[str] | None,
    start: str,
    end: str,
    years_limit: int = 30
):
    """
    Yearly openings (flow) across the given UF set (or all UFs if None).
    - Respects DATE_DT_EXPR parsing.
    - Ignores NULL dates.
    - Normalizes UF filtering via _in_clause_or_all (which compares on UPPER(TRIM(e.uf))).

    Returns a DataFrame with columns: ["year", "openings"] ordered asc.
    """
    years_limit = max(1, int(years_limit))
    uf_filter = _in_clause_or_all(ufs)

    sql = f"""
    WITH base AS (
      SELECT {DATE_DT_EXPR} AS dt
      FROM {fq_table} e
      WHERE 1=1 {uf_filter}
    )
    SELECT
      EXTRACT(YEAR FROM dt)::int AS year,
      COUNT(*)::bigint         AS openings
    FROM base
    WHERE dt IS NOT NULL AND dt BETWEEN '{start}' AND '{end}'
    GROUP BY 1
    ORDER BY 1
    LIMIT {years_limit};
    """
    return _load_df(sql)

@st.cache_data(ttl=300, show_spinner=False)
def df_openings_last_12mo(fq_table: str, uf: str, end: str):
    """
    Monthly openings for the LAST 12 months for a single UF.

    Notes:
    - `end` is the inclusive anchor date in YYYY-MM-DD.
    - We normalize UF with UPPER(TRIM()) and escape quotes.
    - Uses DATE_DT_EXPR to safely coerce text/number dates to DATE.
    - Returns a DataFrame with columns: ["month", "openings"].
    """
    uf_lit = (uf or "").strip().upper().replace("'", "''")
    end_lit = (end or "").strip().replace("'", "''")

    sql = f"""
    WITH bounds AS (
      SELECT date_trunc('month', '{end_lit}'::date) AS end_mo
    ),
    months AS (
      SELECT generate_series(
               (SELECT end_mo FROM bounds) - INTERVAL '11 months',
               (SELECT end_mo FROM bounds),
               INTERVAL '1 month'
             )::date AS month
    ),
    base AS (
      SELECT date_trunc('month', {DATE_DT_EXPR})::date AS month
      FROM {fq_table} e
      WHERE UPPER(TRIM(CAST(e.uf AS TEXT))) = '{uf_lit}'
    )
    SELECT m.month, COALESCE(COUNT(b.month), 0)::bigint AS openings
    FROM months m
    LEFT JOIN base b ON b.month = m.month
    GROUP BY 1
    ORDER BY 1;
    """
    return _load_df(sql)


def plot_openings_monthly(pdf, height: int = 300):
    """
    Simple monthly bar chart for the 'df_openings_last_12mo' output.
    Expects columns: ["month", "openings"].
    """
    fig = go.Figure(go.Bar(
        x=pdf["month"],
        y=pdf["openings"],
        hovertemplate="%{x|%b %Y}<br>New %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=12, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickformat="%b/%y", gridcolor="rgba(255,255,255,.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.08)"),
        transition={"duration": 200, "easing": "cubic-in-out"},
    )
    return fig

@st.cache_data(ttl=300, show_spinner=False)
def df_legal_nature_share(
    fq_table: str,
    ufs: Sequence[str] | None,
    start: str,
    end: str,
    top_n: int = 8
):
    """
    Share of legal natures within the scoped establishments.

    Requirements:
      - A company master table:   empresas or temp_empresas
      - A nature dimension table: natureza_juridica or temp_natureza_juridica

    Approach:
      1) Scope establishments (optionally by UF) and dates using DATE_DT_EXPR.
      2) Collect distinct CNPJ básicos in scope.
      3) Pull each company's natureza_juridica and normalize to a 4-digit code:
         - strip non-digits; if >=4 take first 4; else left-pad to 4; allow NULL.
      4) Build a normalized 4-digit code from the dimension table as well.
      5) Left-join on the code and aggregate counts by description.
      6) Return top N rows ordered by qty desc.

    Returns:
      pandas.DataFrame with columns ["nature", "qty"], or None if dependencies are missing.
    """
    # Resolve supporting tables (final first, then temp); bail out if missing
    emp_tbl = _first_existing_table(["empresas", "temp_empresas"])
    nj_tbl  = _first_existing_table(["natureza_juridica", "temp_natureza_juridica"])
    if not emp_tbl or not nj_tbl:
        return None

    # Build optional UF filter
    uf_filter = _in_clause_or_all(ufs)

    # Hard-escape date literals (defensive)
    start_lit = (start or "").replace("'", "''")
    end_lit   = (end or "").replace("'", "''")

    sql = f"""
    WITH base AS (
      -- Establishments within date range (UF-filtered if provided)
      SELECT {DATE_DT_EXPR} AS dt, e.cnpj_basico
      FROM {fq_table} e
      WHERE 1=1 {uf_filter}
    ),
    scoped AS (
      -- Distinct companies that fall within start/end
      SELECT DISTINCT cnpj_basico
      FROM base
      WHERE dt BETWEEN '{start_lit}' AND '{end_lit}'
    ),
    emp_norm AS (
      -- Normalize the company's natureza_juridica to a 4-digit code
      SELECT
        s.cnpj_basico,
        CASE
          WHEN length(regexp_replace(CAST(emp.natureza_juridica AS TEXT), '\\D', '', 'g')) >= 4
            THEN SUBSTRING(regexp_replace(CAST(emp.natureza_juridica AS TEXT), '\\D', '', 'g') FROM 1 FOR 4)
          ELSE LPAD(NULLIF(regexp_replace(CAST(emp.natureza_juridica AS TEXT), '\\D', '', 'g'), ''), 4, '0')
        END AS code4
      FROM scoped s
      LEFT JOIN {emp_tbl} emp USING (cnpj_basico)
    ),
    dim_norm AS (
      -- Normalize the dimension code the same way
      SELECT
        CASE
          WHEN length(regexp_replace(CAST(nj.codigo AS TEXT), '\\D', '', 'g')) >= 4
            THEN SUBSTRING(regexp_replace(CAST(nj.codigo AS TEXT), '\\D', '', 'g') FROM 1 FOR 4)
          ELSE LPAD(NULLIF(regexp_replace(CAST(nj.codigo AS TEXT), '\\D', '', 'g'), ''), 4, '0')
        END AS code4,
        nj.descricao
      FROM {nj_tbl} nj
    )
    SELECT
      COALESCE(dim_norm.descricao, 'Unknown') AS nature,
      COUNT(*)::bigint AS qty
    FROM emp_norm
    LEFT JOIN dim_norm USING (code4)
    GROUP BY 1
    ORDER BY qty DESC
    LIMIT {int(top_n)};
    """
    try:
        return _load_df(sql)
    except Exception:
        # If anything goes sideways (permissions, schema drift), signal fallback to the caller
        return None

@st.cache_data(ttl=300, show_spinner=False)
def df_status_share(
    fq_table: str,
    ufs: Sequence[str] | None,
    start: str,
    end: str
):
    """
    Active vs. Inactive share for establishments within the given scope.

    Notes
    - '02' in situacao_cadastral is treated as Active; everything else is Inactive.
    - DATE_DT_EXPR is used to robustly coerce dates (works for TEXT and DATE columns).

    Returns
    - pandas.DataFrame with columns: ["status", "qty"]
    """
    uf_filter = _in_clause_or_all(ufs)

    # defensively escape date literals
    start_lit = (start or "").replace("'", "''")
    end_lit   = (end or "").replace("'", "''")

    sql = f"""
    WITH base AS (
      SELECT
        {DATE_DT_EXPR} AS dt,
        COALESCE(LPAD(TRIM(CAST(e.situacao_cadastral AS TEXT)), 2, '0'), '') AS sit
      FROM {fq_table} e
      WHERE 1=1 {uf_filter}
    )
    SELECT
      CASE WHEN sit = '02' THEN 'Active' ELSE 'Inactive' END AS status,
      COUNT(*)::bigint AS qty
    FROM base
    WHERE dt BETWEEN '{start_lit}' AND '{end_lit}'
    GROUP BY 1
    ORDER BY 2 DESC;
    """
    return _load_df(sql)

@st.cache_data(ttl=300, show_spinner=False)
def df_cnae_profile(
    fq_table: str,
    uf: str,
    start: str,
    end: str,
    top_n: int = 10
):
    """
    CNAE mix for a single UF, grouped by 2-digit CNAE division.

    Column selection strategy
    - Prefer 'cnae_fiscal_principal'; fall back to: 'cnae_fiscal', 'cnae', 'cnae_principal'.
    - If none exists, return None.

    Returns
    - pandas.DataFrame with columns: ["division2", "qty"], or None if no CNAE column is available.
    """
    # pick the first usable CNAE column
    eng = _engine()
    candidates = ["cnae_fiscal_principal", "cnae_fiscal", "cnae", "cnae_principal"]
    chosen_col: Optional[str] = None
    for col in candidates:
        try:
            with eng.connect() as c:
                # lightweight probe; will fail fast if column doesn't exist
                c.execute(text(f"SELECT {col} FROM {fq_table} LIMIT 1"))
            chosen_col = col
            break
        except Exception:
            continue

    if not chosen_col:
        return None

    # escape literals
    uf_lit     = (uf or "").replace("'", "''")
    start_lit  = (start or "").replace("'", "''")
    end_lit    = (end or "").replace("'", "''")

    # compute mix by 2-digit CNAE division
    sql = f"""
    WITH base AS (
      SELECT
        {DATE_DT_EXPR} AS dt,
        REGEXP_REPLACE(CAST(e.{chosen_col} AS TEXT), '\\D', '', 'g') AS cnae
      FROM {fq_table} e
      WHERE e.uf = '{uf_lit}'
    ),
    filtered AS (
      SELECT SUBSTRING(cnae, 1, 2) AS division2
      FROM base
      WHERE dt BETWEEN '{start_lit}' AND '{end_lit}'
        AND cnae IS NOT NULL
        AND length(cnae) >= 2
    )
    SELECT division2, COUNT(*)::bigint AS qty
    FROM filtered
    GROUP BY 1
    ORDER BY qty DESC
    LIMIT {int(top_n)};
    """
    return _load_df(sql)

@st.cache_data(ttl=300, show_spinner=False)
def df_enterprise_profile_by_uf(
    fq_table: str,
    uf: str,
    start: str,
    end: str,
):
    """
    Enterprise profile for a single UF:
      - total companies in scope
      - MEI count and density
      - average (raw) capital
      - size/porte mix (micro/small/medium/large)

    Implementation details
    - Looks for a final table first, then temp_ fallback:
        * companies:   ["empresas", "temp_empresas"]
        * simples/MEI: ["simples",  "temp_simples"]
    - If companies table is missing, returns None (cannot compute profile).
    - If simples table is missing, MEI is treated as zero safely.
    - Dates are coerced using DATE_DT_EXPR (works for TEXT/DATE).
    """
    # Resolve dimension tables
    emp_tbl = _first_existing_table(["empresas", "temp_empresas"])
    simples_tbl = _first_existing_table(["simples", "temp_simples"])

    if not emp_tbl:
        return None  # cannot compute without company attributes

    # Escape user-provided literals
    uf_lit = (uf or "").replace("'", "''")
    start_lit = (start or "").replace("'", "''")
    end_lit = (end or "").replace("'", "''")

    # Build MEI CTE depending on simples availability to avoid "None" in SQL
    if simples_tbl:
        mei_cte = f"""
        , mei AS (
          SELECT s.cnpj_basico,
                 CASE WHEN sm.opcao_mei = 'S' THEN 1 ELSE 0 END AS is_mei
          FROM scoped s
          LEFT JOIN {simples_tbl} sm USING (cnpj_basico)
        )
        """
        mei_join = "LEFT JOIN mei USING (cnpj_basico)"
        mei_sum_expr = "SUM(CASE WHEN mei.is_mei = 1 THEN 1 ELSE 0 END)::bigint AS n_mei"
    else:
        # No simples table → zero MEI but keep the same SELECT signature
        mei_cte = """
        , mei AS (
          SELECT cnpj_basico, 0::int AS is_mei
          FROM scoped
        )
        """
        mei_join = "LEFT JOIN mei USING (cnpj_basico)"
        mei_sum_expr = "SUM(0)::bigint AS n_mei"

    sql = f"""
    WITH base AS (
      SELECT {DATE_DT_EXPR} AS dt, e.uf, e.cnpj_basico
      FROM {fq_table} e
      WHERE e.uf = '{uf_lit}'
    ),
    scoped AS (
      SELECT DISTINCT cnpj_basico
      FROM base
      WHERE dt BETWEEN '{start_lit}' AND '{end_lit}'
    ),
    emp AS (
      SELECT
        s.cnpj_basico,
        emp.natureza_juridica::text AS nat_raw,
        emp.porte_empresa::text     AS size_raw,
        NULLIF(REPLACE(emp.capital_social, ',', '.'), '')::numeric AS cap
      FROM scoped s
      LEFT JOIN {emp_tbl} emp USING (cnpj_basico)
    )
    {mei_cte}
    SELECT
      COUNT(*)::bigint AS n_emp,
      {mei_sum_expr},
      AVG(cap) AS avg_cap_raw,
      -- Size buckets (tune mappings to your dictionary if needed)
      SUM(CASE WHEN size_raw IN ('01','1','ME','M')  THEN 1 ELSE 0 END)::bigint AS micro,
      SUM(CASE WHEN size_raw IN ('02','2','EPP','P') THEN 1 ELSE 0 END)::bigint AS small,
      SUM(CASE WHEN size_raw IN ('03','3')           THEN 1 ELSE 0 END)::bigint AS medium,
      SUM(CASE WHEN size_raw IN ('05','5','04','4')  THEN 1 ELSE 0 END)::bigint AS large
    FROM emp
    {mei_join};
    """
    return _load_df(sql)

@st.cache_data(ttl=300, show_spinner=False)
def df_data_quality(fq_table: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Basic data-quality coverage for the establishments table:
      - total rows
      - rows missing UF
      - rows missing municipality
      - rows with missing/invalid start date

    Returns
    -------
    (total, missing_uf, missing_municipio, missing_date) or None on failure.
    """
    sql = f"""
    WITH raw AS (
      SELECT
        CASE
          WHEN COALESCE(NULLIF(TRIM(CAST(uf AS TEXT)), ''), '') = '' THEN 1 ELSE 0
        END AS miss_uf,
        CASE
          WHEN COALESCE(NULLIF(TRIM(CAST(municipio AS TEXT)), ''), '') = '' THEN 1 ELSE 0
        END AS miss_mun,
        CASE
          WHEN NULLIF(REGEXP_REPLACE(CAST(data_inicio_atividades AS TEXT), '\\D', '', 'g'), '') IS NULL THEN 1
          WHEN LENGTH(REGEXP_REPLACE(CAST(data_inicio_atividades AS TEXT), '\\D', '', 'g')) < 8 THEN 1
          ELSE 0
        END AS miss_dt
      FROM {fq_table}
    )
    SELECT
      COUNT(*)::bigint AS total,
      SUM(miss_uf)::bigint  AS miss_uf,
      SUM(miss_mun)::bigint AS miss_mun,
      SUM(miss_dt)::bigint  AS miss_dt
    FROM raw;
    """
    try:
        df = _load_df(sql)
        if df is None or df.empty:
            return None
        row = df.iloc[0]
        return (
            int(row["total"]),
            int(row["miss_uf"]),
            int(row["miss_mun"]),
            int(row["miss_dt"]),
        )
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def df_last_available_month(fq_table: str) -> Optional[str]:
    """
    Returns the latest month (YYYY-MM) present in the establishments table,
    using the robust DATE_DT_EXPR to coerce dates from TEXT/DATE.
    """
    sql = f"SELECT MAX({DATE_DT_EXPR}) AS max_dt FROM {fq_table} e;"
    try:
        df = _load_df(sql)
        if df is None or df.empty:
            return None
        mx = df.iloc[0]["max_dt"]
        return None if mx is None else str(mx)[:7]  # "YYYY-MM"
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def get_active_rate(fq_table: str, start: str, end: str) -> float:
    """
    Return the percentage of 'active' establishments (situacao_cadastral == '02')
    within the given date range [start, end], using robust date coercion.
    """
    sql = f"""
    WITH base AS (
      SELECT
        COALESCE(LPAD(TRIM(CAST(e.situacao_cadastral AS TEXT)), 2, '0'), '') AS sit,
        {DATE_DT_EXPR} AS dt
      FROM {fq_table} e
    )
    SELECT
      SUM(CASE WHEN sit = '02' THEN 1 ELSE 0 END)::bigint AS active,
      COUNT(*)::bigint AS total
    FROM base
    WHERE dt BETWEEN '{start}' AND '{end}';
    """
    try:
        df = _load_df(sql)
        if df is None or df.empty:
            return 0.0
        active = int(df.iloc[0]["active"] or 0)
        total  = int(df.iloc[0]["total"] or 0)
        return (active / total * 100.0) if total else 0.0
    except Exception:
        return 0.0

@st.cache_data(ttl=300, show_spinner=False)
def df_top5_concentration(
    fq_table: str,
    uf: str,
    start: str,
    end: str
) -> Optional[Tuple[int, int]]:
    """
    Concentration index helper within a UF.
    Returns a pair: (sum of top-5 municipalities, total of the UF) for the period.
    If the municipality dimension table is unavailable or a query fails, returns None.
    """
    # Resolve municipality dimension table (final first, then temp)
    mun_tbl = _first_existing_table(["municipio", "temp_municipio"])
    if not mun_tbl:
        return None

    # Escape user-provided literals to avoid breaking SQL
    uf_lit = (uf or "").replace("'", "''")
    start_lit = (start or "").replace("'", "''")
    end_lit = (end or "").replace("'", "''")

    sql = f"""
    WITH base AS (
      SELECT {DATE_DT_EXPR} AS dt, e.uf, e.municipio
      FROM {fq_table} e
      WHERE e.uf = '{uf_lit}'
    ),
    inrng AS (
      SELECT municipio
      FROM base
      WHERE dt BETWEEN '{start_lit}' AND '{end_lit}'
    ),
    agg AS (
      SELECT m.descricao AS municipio, COUNT(*)::bigint AS qty
      FROM inrng i
      LEFT JOIN {mun_tbl} m ON m.codigo = i.municipio
      WHERE m.descricao IS NOT NULL
      GROUP BY 1
    ),
    ordered AS (
      SELECT municipio, qty,
             ROW_NUMBER() OVER (ORDER BY qty DESC) AS rn
      FROM agg
    )
    SELECT
      (SELECT COALESCE(SUM(qty), 0) FROM ordered WHERE rn <= 5) AS top5,
      (SELECT COALESCE(SUM(qty), 0) FROM ordered)              AS total;
    """
    try:
        df = _load_df(sql)
        if df is None or df.empty:
            return None
        r = df.iloc[0]
        return int(r["top5"]), int(r["total"])
    except Exception:
        return None

# ───────────────────────────── Geometry helpers ─────────────────────────────
from typing import Iterable, List, Tuple

# Fallback center roughly over Brazil (lon, lat)
_BR_FALLBACK_CENTER: Tuple[float, float] = (-53.1, -14.2)


def _flatten_coords(coords: Iterable) -> List[Tuple[float, float]]:
    """
    Recursively flatten a nested coordinate structure (Polygon or MultiPolygon)
    into a simple list of (lon, lat) tuples.

    Assumptions:
      - Coordinates follow GeoJSON order: [longitude, latitude].
      - `coords` may contain multiple nesting levels: e.g.
        MultiPolygon -> list[Polygon] -> list[LinearRing] -> list[Position].

    Returns:
      A list of (lon, lat) float pairs. Invalid points are skipped.
    """
    points: List[Tuple[float, float]] = []

    if not isinstance(coords, (list, tuple)):
        return points  # nothing to do

    # Base case: a single position like [lon, lat]
    # Guard against short or non-numeric entries.
    if coords and isinstance(coords[0], (int, float)):
        try:
            lon = float(coords[0])
            lat = float(coords[1])  # may raise IndexError if malformed
            points.append((lon, lat))
        except Exception:
            pass
        return points

    # Recursive case: nested lists/tuples
    for item in coords:
        points.extend(_flatten_coords(item))

    return points


def _centroid(feature: dict) -> Tuple[float, float]:
    """
    Compute a simple centroid (mean of vertices) for a GeoJSON Feature
    whose geometry type is Polygon or MultiPolygon.

    Notes:
      - This is a quick visual centroid, NOT area-weighted and NOT geodesic.
      - Good enough for label placement; do NOT use for precise geospatial work.
      - Falls back to a fixed point over Brazil if geometry is missing or invalid.

    Args:
      feature: GeoJSON Feature dict with "geometry": {"type": "...", "coordinates": ...}

    Returns:
      (lon, lat) tuple for the approximate centroid.
    """
    if not isinstance(feature, dict):
        return _BR_FALLBACK_CENTER

    geom = feature.get("geometry") or {}
    gtype = geom.get("type")
    coords = geom.get("coordinates")

    if gtype not in {"Polygon", "MultiPolygon"} or coords is None:
        return _BR_FALLBACK_CENTER

    pts = _flatten_coords(coords)
    if not pts:
        return _BR_FALLBACK_CENTER

    # Arithmetic mean of all vertices
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

# ───────────────────────────────── Plots ───────────────────────────────────────
from typing import Sequence, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def build_map(
    df: pd.DataFrame,
    geo: dict,
    selected: Optional[Sequence[str]] = None,
    *,
    height: int = 560,
    zoom: float = 3.1,
    center: dict = {"lat": -14.5, "lon": -53.5},
) -> go.Figure:
    """
    Build a choropleth mapbox of Brazilian states (UF) with:
      - heat scale on company counts,
      - thin white boundaries,
      - static UF labels at centroid,
      - optional red overlay for a set of selected UFs (multi-select).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
          - 'uf'  : state code (e.g., 'SP')
          - 'qty' : integer count for the heat color.
    geo : dict
        GeoJSON FeatureCollection with features that include properties.sigla.
    selected : sequence of str or None
        UFs to be highlighted in red as a translucent overlay.
    height : int
        Figure height in pixels.
    zoom : float
        Initial zoom for the mapbox camera.
    center : dict
        Map center, e.g. {"lat": -14.5, "lon": -53.5}.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # ── Guard clauses and light normalization
    if df is None or df.empty:
        # Return an empty figure with consistent layout rather than error
        fig = go.Figure()
        fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0))
        return fig

    if "uf" not in df.columns or "qty" not in df.columns:
        raise ValueError("DataFrame must contain 'uf' and 'qty' columns.")

    # Avoid degenerate color ranges (e.g., when min == max)
    vmin = int(df["qty"].min()) if len(df) else 0
    vmax = int(df["qty"].max()) if len(df) else 1
    if vmin == vmax:
        # Expand a little so the colorbar still shows a gradient
        vmin = max(0, vmin - 1)
        vmax = vmax + 1

    selected_set = set(selected or [])

    # ── Base choropleth layer
    fig = px.choropleth_mapbox(
        df,
        geojson=geo,
        locations="uf",
        featureidkey="properties.sigla",
        color="qty",
        color_continuous_scale=[THEME["heat0"], THEME["heat1"], THEME["heat2"], THEME["heat3"], THEME["heat4"]],
        range_color=[vmin, vmax],
        mapbox_style="carto-darkmatter",
        zoom=zoom,
        center=center,
        opacity=0.92,
        labels={"qty": "Establishments", "uf": "UF"},
    )

    # Subtle white borders around UFs
    fig.update_traces(
        hovertemplate="<b>%{location}</b><br>Establishments: %{z:,}<extra></extra>",
        marker_line_width=0.9,
        marker_line_color="rgba(255,255,255,.48)",
    )

    # ── Static UF labels at centroid (single scatter trace, no hover)
    lons, lats, texts = [], [], []
    for f in geo.get("features", []):
        props = f.get("properties", {}) or {}
        uf = props.get("sigla")
        if not uf:
            continue
        lon, lat = _centroid(f)
        lons.append(lon)
        lats.append(lat)
        texts.append(uf)

    if lons:  # add only if we found valid centroids
        fig.add_trace(
            go.Scattermapbox(
                lon=lons,
                lat=lats,
                mode="text",
                text=texts,
                textfont=dict(size=13, color="#E6EDFF"),
                hoverinfo="skip",
                name="UF labels",
            )
        )

    # ── Selected overlay (multi) — translucent red via a separate choropleth
    geo_siglas = {f.get("properties", {}).get("sigla") for f in geo.get("features", [])}
    sel_locs = [u for u in selected_set if u in geo_siglas]

    if sel_locs:
        sel_features = [f for f in geo.get("features", []) if f.get("properties", {}).get("sigla") in sel_locs]
        sel_geo = {"type": "FeatureCollection", "features": sel_features}
        highlight = "rgba(255,107,107,0.55)"  # theme red with alpha

        fig.add_trace(
            go.Choroplethmapbox(
                geojson=sel_geo,
                locations=sel_locs,
                z=[1] * len(sel_locs),  # dummy z for colorscale
                zmin=0,
                zmax=1,
                featureidkey="properties.sigla",
                showscale=False,
                colorscale=[[0, highlight], [1, highlight]],
                hoverinfo="skip",
                name="Selected",
            )
        )

    # ── Final layout polish (thin outline layer below traces, tidy colorbar, etc.)
    fig.update_layout(
        mapbox=dict(pitch=45, bearing=-15),
        mapbox_layers=[
            {
                "source": geo,
                "type": "line",
                "below": "traces",
                "color": "rgba(255,255,255,.28)",
                "line": {"width": 1.4},
            }
        ],
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            title="Qty",
            thickness=12,
            len=1.0,
            y=0.5,
            yanchor="middle",
            outlinewidth=0,
            tickcolor="#9AA6BF",
            bgcolor="rgba(0,0,0,.35)",
        ),
        hoverlabel=dict(
            bgcolor="rgba(13,20,38,.98)",
            font_size=13,
            font_color=THEME["text"],
            bordercolor="rgba(255,255,255,.15)",
        ),
        transition={"duration": 180, "easing": "cubic-in-out"},
        dragmode="pan",
        showlegend=False,
    )

    return fig

def donut_pie(
    df: pd.DataFrame,
    names_col: str,
    values_col: str,
    *,
    height: int = 560,
    color_map: Optional[Mapping[str, str]] = None,
    sequence: Optional[Sequence[str]] = None,
    show_values: bool = False,
) -> go.Figure:
    """
    Build a donut pie with sensible defaults, safe fallbacks, and crisp hover.

    Parameters
    ----------
    df : DataFrame with at least `names_col` and `values_col`
    names_col : column for slice labels
    values_col : column for slice values
    height : figure height in px
    color_map : explicit color mapping per label (overrides `sequence`)
    sequence : color sequence fallback (used if `color_map` is None)
    show_values : if True, shows absolute values alongside percentages
    """
    # Guard: empty input → minimalist placeholder
    if df is None or df.empty or names_col not in df.columns or values_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8))
        return fig

    # Ensure non-negative numeric values and drop zero/NaN safely
    work = df[[names_col, values_col]].copy()
    work[values_col] = pd.to_numeric(work[values_col], errors="coerce").fillna(0)
    work = work[work[values_col] > 0]
    if work.empty:
        fig = go.Figure()
        fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8))
        return fig

    # Build pie
    if color_map:
        fig = px.pie(
            work,
            names=names_col,
            values=values_col,
            hole=0.58,
            color=names_col,
            color_discrete_map=color_map,
        )
    else:
        fig = px.pie(
            work,
            names=names_col,
            values=values_col,
            hole=0.58,
            color=names_col,
            color_discrete_sequence=sequence,
        )

    # Trace styling
    if show_values:
        # percent + absolute values in the label text
        textinfo = "percent+value"
    else:
        textinfo = "percent"

    pulls = [0.03] + [0.02] * max(0, len(work) - 1)
    fig.update_traces(
        textinfo=textinfo,
        textfont=dict(size=12),
        hovertemplate="%{label}: %{percent} (%{value:,})<extra></extra>",
        pull=pulls,
        marker=dict(line=dict(color="rgba(0,0,0,.25)", width=0.5)),
        sort=False,  # respect input order when desired
    )

    # Layout polish
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=12),
            itemwidth=30,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        transition={"duration": 180, "easing": "cubic-in-out"},
    )
    return fig


def openings_area(
    df: pd.DataFrame,
    *,
    height: int = 420,
    line_color: Optional[str] = None,
    fill_rgba: Optional[str] = None,
) -> go.Figure:
    """
    Area/line chart for yearly openings. Robust to single-point series and unsorted input.

    Parameters
    ----------
    df : DataFrame with columns ['year', 'openings']
    height : figure height in px
    line_color : optional hex/rgba color for line/markers (defaults to THEME["aqua"])
    fill_rgba : optional rgba for area fill (defaults to a soft aqua fill)
    """
    # Guard: empty or missing columns
    if df is None or df.empty or not {"year", "openings"}.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(height=height, margin=dict(l=10, r=12, t=6, b=10))
        return fig

    work = df[["year", "openings"]].copy()
    # Coerce and clean
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["openings"] = pd.to_numeric(work["openings"], errors="coerce").fillna(0)
    work = work.dropna(subset=["year"]).sort_values("year")
    if work.empty:
        fig = go.Figure()
        fig.update_layout(height=height, margin=dict(l=10, r=12, t=6, b=10))
        return fig

    # Defaults tied to your theme
    lc = line_color or THEME["aqua"]
    fr = fill_rgba or "rgba(23,184,144,.22)"  # soft aqua fill

    # Build figure
    fig = go.Figure(
        go.Scatter(
            x=work["year"],
            y=work["openings"],
            mode="lines+markers",
            line=dict(width=3, color=lc),
            marker=dict(size=7, color=lc),
            fill="tozeroy",
            fillcolor=fr,
            hovertemplate="Year %{x:.0f}<br>New %{y:,}<extra></extra>",
        )
    )

    # If single point, keep markers only (avoid a tiny “area” blob)
    if len(work) == 1:
        fig.update_traces(mode="markers", fill=None)

    # Layout polish
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=12, t=6, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=None,
            gridcolor="rgba(255,255,255,.04)",
            zeroline=False,
            tickformat="d",
        ),
        yaxis=dict(
            title=None,
            gridcolor="rgba(255,255,255,.06)",
            zeroline=False,
            tickformat=",",  # thousands separator
        ),
        transition={"duration": 180, "easing": "cubic-in-out"},
    )
    return fig

# ───────────────────────────── Enhanced summary table (below map) ─────────────────────────────
def render_state_summary(uf_df, selected_set) -> None:
    """
    Display a compact, interactive table that helps interpret the choropleth:

    - Rank, Region, Coastal/Interior tag, absolute totals and share of Brazil
    - Controls: Top-N slider, "Only selected" toggle, Region filter, quick UF search
    - CSV download with raw values (no string formatting)

    Notes:
    - Works with an empty `selected_set` (interpreted as “all Brazil”).
    - If `uf_df` is empty, a friendly info message is shown instead of raising.
    """
    with block():
        st.markdown("#### State summary")

        if uf_df is None or uf_df.empty:
            st.info("No data to display.")
            return

        # Normalize selection and compute helpers
        selected_set = set(selected_set or [])
        total_brazil = int(uf_df["qty"].sum())

        df = uf_df.copy()
        df["Rank"] = df["qty"].rank(method="dense", ascending=False).astype(int)
        df["% of total"] = (df["qty"] / total_brazil * 100).round(2) if total_brazil else 0.0
        df["Selected"] = df["uf"].isin(selected_set)
        df["Region"] = df["uf"].map(UF_TO_REGION)
        df["Coastal"] = df["uf"].isin(COASTAL_UF)

        # ── Controls
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
        with c1:
            max_n = len(df)
            default_n = min(15, max_n if max_n > 0 else 15)
            top_n = st.slider("Top-N", min_value=5, max_value=max(5, max_n), value=default_n, key="tbl_topn_geo")
        with c2:
            only_sel = st.checkbox("Only selected", value=False, key="tbl_only_sel_geo")
        with c3:
            reg = st.selectbox("Region", ["All", "N", "NE", "CO", "SE", "S"], index=0, key="tbl_region")
        with c4:
            q = st.text_input("Search UF", value="", placeholder="e.g., SP, RJ…", key="tbl_search_geo").strip().upper()

        # ── Filters
        view = df
        if only_sel:
            view = view[view["Selected"]]
        if reg != "All":
            view = view[view["Region"] == reg]
        if q:
            view = view[view["uf"].astype(str).str.upper().str.contains(q, na=False)]

        # Sort: selected first, then by qty desc; apply Top-N
        view = view.sort_values(["Selected", "qty"], ascending=[False, False]).head(top_n)

        if view.empty:
            st.info("No states match the current filters.")
            return

        # ── Pretty view
        show = (
            view.rename(columns={"uf": "UF", "qty": "Establishments"})
                .assign(
                    **{
                        "Establishments": view["qty"].map(fmt_num),
                        "% of total": view["% of total"].map(lambda x: f"{x:.2f}%"),
                        "Selected": view["Selected"].map({True: "✓", False: ""}),
                        "Coastal": view["Coastal"].map({True: "Coastal", False: "Interior"}),
                    }
                )
        )

        st.dataframe(
            show[["Rank", "UF", "Region", "Coastal", "Establishments", "% of total", "Selected"]],
            hide_index=True,
            use_container_width=True,
        )

        # ── CSV download (raw values, no string formatting)
        raw = view[["Rank", "uf", "Region", "Coastal", "qty", "% of total", "Selected"]].rename(
            columns={"uf": "UF", "qty": "Establishments", "% of total": "Share_percent"}
        )
        st.download_button(
            "Download CSV",
            data=raw.to_csv(index=False).encode("utf-8"),
            file_name="state_summary.csv",
            mime="text/csv",
            use_container_width=True,
            key="tbl_download_csv",
        )


# ───────────────────────────────────────── Header & global date filter 
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:12px;">
      <span style="font-size:28px">🗺️</span>
      <div style="font-size:22px;font-weight:900;letter-spacing:.2px;">
        Geography — Establishments by State & City
      </div>
    </div>
    <div style="color:var(--muted);font-size:13px;">
      <b>Map = stock</b> (where companies are). <b>Lines = flow</b> (new openings over time).
      Click states to multi-select; empty selection = Brazil.
    </div>
    """,
    unsafe_allow_html=True,
)

# Global date filter (fast, compact; all top-level metrics respect this range)
start_str, end_str = date_filter_compact(default_months=12)

# ───────────────────────────────── Data load ───────────────────────────────────
# Load static geometry
geo = load_geojson()

# Discover the primary (establishments) fact table and supporting dimensions.
# We always prefer the final table name; if not present, we fall back to temp_* across all user schemas.
fq_table    = detect_estab_table(debug=False)
EMP_TBL     = _first_existing_table(["empresas", "temp_empresas"])
SIMPLES_TBL = _first_existing_table(["simples", "temp_simples"])
NATJUR_TBL  = _first_existing_table(["natureza_juridica", "temp_natureza_juridica"])
MUNIC_TBL   = _first_existing_table(["municipio", "temp_municipio"])

# Hard stop if the fact table is missing (no point in rendering the page)
if not fq_table:
    st.error("Could not find a non-empty establishments table.")
    st.stop()

# Load UF-level stock (respecting the global date filter set above)
uf_df = df_uf_stock(fq_table, start=start_str, end=end_str)
if uf_df is None or uf_df.empty:
    st.warning("No data for the current period.")
    st.stop()

# Optional: lightweight diagnostics if key dimension tables are missing.
# We keep the app running and gracefully degrade specific widgets that depend on them.
missing_dims = []
if EMP_TBL is None:     missing_dims.append("companies (empresas)")
if SIMPLES_TBL is None: missing_dims.append("Simples/MEI (simples)")
if NATJUR_TBL is None:  missing_dims.append("legal nature (natureza_juridica)")
if MUNIC_TBL is None:   missing_dims.append("municipalities (municipio)")
if missing_dims:
    st.caption("⚠️ Limited functionality: missing dimension(s) — " + ", ".join(missing_dims))

# Selection model (multi-select via map or widget)
# Empty set = “Brazil (all states)”
if "uf_selected_set" not in st.session_state:
    st.session_state.uf_selected_set = set()

all_ufs = uf_df["uf"].tolist()
selected_set: set[str] = set(st.session_state.uf_selected_set)


# ───────────────────────────── KPIs (national / regional) ──────────────────────
# Scope = all Brazil when nothing is selected; otherwise filter to selected UFs.
scope_df = uf_df[uf_df["uf"].isin(selected_set)] if selected_set else uf_df

# Guard against unexpected empties (uf_df already validated, but keep this safe)
if scope_df is None or scope_df.empty:
    st.warning("No rows in scope for the selected filters.")
    st.stop()

# Totals and leader UF within the current scope
total_count = int(scope_df["qty"].sum())
leader_row = scope_df.iloc[0]
leader_uf  = str(leader_row["uf"])
leader_val = int(leader_row["qty"])

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    # Headline KPI: total companies in scope
    title = "Selected — total companies" if selected_set else "Brazil — total companies"
    sub = ", ".join(sorted(selected_set)) if selected_set else f"Period: {start_str} → {end_str}"
    kpi_card(
        title,
        f'<span class="accent">{fmt_num(total_count)}</span>',
        sub=sub,
        variant=1,
    )

with c2:
    # Largest contributor within the scope
    kpi_card(
        "Largest UF (stock)",
        f'<span class="accent-b">{fmt_num(leader_val)}</span>',
        sub=f"UF: <b>{leader_uf}</b>",
        variant=2,
    )

with c3:
    active_pct = get_active_rate(fq_table, start_str, end_str)
    kpi_card("% Active", f"{active_pct:.1f}%", sub=f"Period: {start_str} → {end_str}", variant=3)

hr()

# Compact coverage note (freshness + misses), as a caption under KPIs
_last = df_last_available_month(fq_table) or "unknown"
_dq = df_data_quality(fq_table)
if _dq:
    tot, muf, mmun, mdt = _dq
    st.caption(
        f"Dataset coverage — Up to **{_last}** • "
        f"UF miss {muf/tot*100:.1f}% • Mun miss {mmun/tot*100:.1f}% • Date miss {mdt/tot*100:.1f}%"
    )
else:
    st.caption(f"Dataset coverage — Up to **{_last}**")


# ───────────────────────────── Map (left) + Donut (right) ─────────────────────
left, right = st.columns([1.6, 1], gap="large")

# ——————————————— Left: Interactive choropleth (multi-select by click) ———————————————
with left:
    with block():
        fig_map = build_map(
            uf_df,
            geo,
            sorted(selected_set) if selected_set else None
        )

        clicked_state = None
        used_events = False

        # Try to capture Plotly click events (falls back to a static chart gracefully)
        try:
            from streamlit_plotly_events import plotly_events  # type: ignore
            used_events = True
            events = plotly_events(
                fig_map,
                click_event=True,
                hover_event=False,
                select_event=False,
                key="map_click_multi"
            )
            if events:
                # Location code lives either in "location" or inside "pointData"
                clicked_state = (
                    events[0].get("location")
                    or events[0].get("pointData", {}).get("location")
                )
        except Exception:
            # Fallback: just render the map if the events package is not available
            pass

        # Always render the map once (avoid double rendering)
        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

        # Toggle selection & refresh immediately to repaint + re-filter
        if clicked_state:
            if clicked_state in selected_set:
                selected_set.remove(clicked_state)
            else:
                selected_set.add(clicked_state)
            st.session_state.uf_selected_set = selected_set
            st.rerun()

# ——————————————— Right: Controls + “Legal nature / Status” donut ———————————————
with right:
    # Multiselect kept in sync with map clicks (empty = all UFs)
    current = (
        sorted(list(selected_set), key=lambda x: all_ufs.index(x))
        if selected_set else []
    )
    chosen = st.multiselect(
        "States (UF)",
        options=all_ufs,
        default=current,
        placeholder="All states"
    )
    new_set = set(chosen)
    if new_set != selected_set:
        st.session_state.uf_selected_set = new_set
        st.rerun()

    # Donut: “Legal nature — share (top)” with safe fallback to “Status share”
    with block():
        st.markdown("#### Legal nature — share (top)")
        active_set = sorted(st.session_state.uf_selected_set) if st.session_state.uf_selected_set else None
        top_nat = df_legal_nature_share(
            fq_table,
            active_set,
            start=start_str, end=end_str, top_n=8
        )


        if top_nat is not None and not top_nat.empty:
            # Rename columns for nice legend labels
            donut_fig = donut_pie(
                top_nat.rename(columns={"nature": "Legal nature", "qty": "qty"}),
                names_col="Legal nature",
                values_col="qty",
                sequence=[THEME["aqua"], THEME["hero1_b"], THEME["violet"], THEME["red"], THEME["heat4"], THEME["heat3"]],
                height=560
            )
            st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown(
                "<div style='margin:-2px 0 8px 0;color:var(--muted);font-size:12px;'>"
                "Legal-nature dimension unavailable → showing status share."
                "</div>",
                unsafe_allow_html=True
            )
            status_df = df_status_share(
            fq_table,
            active_set,
            start=start_str,
            end=end_str
            )

            if status_df is not None and not status_df.empty:
                status_fig = donut_pie(
                    status_df.rename(columns={"status": "Status", "qty": "qty"}),
                    names_col="Status",
                    values_col="qty",
                    color_map={"Active": THEME["aqua"], "Inactive": THEME["red"]},
                    height=560
                )
                st.plotly_chart(status_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No status breakdown for this scope/period.")

# ——————————————— Table: quick state summary under the map ———————————————
render_state_summary(uf_df, st.session_state.uf_selected_set)

hr()

# ───────────────────────────── Municipal drill-down (clicked UF) ──────────────────────────
with block():
    st.markdown("### Cities within the selected state (UF)")

    # Require at least one UF selected
    if not st.session_state.uf_selected_set:
        st.info("Tip: select exactly one UF on the map (or use the multiselect) to view its cities.")
    else:
        # If multiple UFs are selected, let the user choose which one to drill down into
        chosen_pool = sorted(list(st.session_state.uf_selected_set))
        if len(chosen_pool) > 1:
            focus_uf = st.selectbox(
                "Choose a UF to drill down",
                options=chosen_pool,
                index=0,
                key="muni_drill_select"
            )
        else:
            focus_uf = chosen_pool[0]

        st.markdown(f"**Focused UF:** `{focus_uf}`")

        # Pull city ranking for the focused UF
        cities = df_municipios_in_uf(
            fq_table,
            focus_uf,
            start=start_str,
            end=end_str,
            limit_n=100
        )

        if cities is None or cities.empty:
            st.info("No city breakdown available for the selected UF in this period.")
        else:
            # Headline KPIs
            uf_total = int(cities["qty"].sum())
            top_city = str(cities.iloc[0]["municipio"])
            top_val  = int(cities.iloc[0]["qty"])
            top_pct  = (top_val / uf_total * 100) if uf_total else 0.0

            k1, k2, k3 = st.columns(3)
            with k1:
                kpi_card("UF total (stock)", fmt_num(uf_total), sub=f"UF: {focus_uf}", variant=1)
            with k2:
                kpi_card("Top city (stock)", fmt_num(top_val), sub=top_city, variant=2)
            with k3:
                kpi_card("Top city share", f"{top_pct:.2f}%", sub="of UF total", variant=3)

            # Controls: Top-N and search
            c1, c2 = st.columns([1, 1.2])
            with c1:
                top_n = st.slider(
                    "Show top-N cities",
                    min_value=5,
                    max_value=min(50, len(cities)),
                    value=min(15, len(cities)),
                    key="muni_topn"
                )
            with c2:
                q = st.text_input(
                    "Filter by city name",
                    value="",
                    placeholder="Type part of a city name…",
                    key="muni_search"
                ).strip()

            # Apply filters for the table view
            table_df = cities.copy()
            if q:
                table_df = table_df[table_df["municipio"].astype(str).str.contains(q, case=False, na=False)]

            table_df = table_df.sort_values("qty", ascending=False).head(top_n)

            # Horizontal bar chart for top cities (always uses the top-N ordering)
            chart_df = cities.sort_values("qty", ascending=False).head(top_n)
            bar = px.bar(
                chart_df,
                x="qty",
                y="municipio",
                orientation="h",
                labels={"qty": "Companies", "municipio": "City"},
                text_auto=True
            )
            bar.update_layout(
                height=420,
                margin=dict(l=8, r=8, t=8, b=8),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(categoryorder="total ascending"),
                transition={"duration": 200, "easing": "cubic-in-out"}
            )
            st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

            # Ranked table (friendly formatting)
            view = table_df.copy()
            view["Rank"] = view["qty"].rank(method="dense", ascending=False).astype(int)
            view = view.sort_values(["qty"], ascending=False)
            view = view.rename(columns={"municipio": "City", "qty": "Companies"})
            view["Companies"] = view["Companies"].map(fmt_num)

            st.dataframe(
                view[["Rank", "City", "Companies"]],
                hide_index=True,
                use_container_width=True
            )

            # CSV download (raw numbers)
            raw = table_df.rename(columns={"municipio": "city", "qty": "companies"})
            st.download_button(
                "Download city ranking (CSV)",
                data=raw.to_csv(index=False).encode("utf-8"),
                file_name=f"cities_{focus_uf.lower()}_{start_str}_to_{end_str}.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Helper text for context
            with st.expander("What this shows", expanded=False):
                st.markdown(
                    "- **Stock** = how many companies are registered in each city within this UF "
                    f"during the selected period ({start_str} → {end_str}).\n"
                    "- **Top city share** highlights concentration inside the UF.\n"
                    "- Use **Top-N** and **Filter by city name** to refine the table and chart."
                )

hr()

# ───────────────────────────── Time + Geography (narrative) ─────────────────────────────
with block():
    # Left: title + context. Right: a local (independent) date filter for the yearly trend.
    lc, rc = st.columns([1.0, 1.0])

    with lc:
        scope_lbl = (
            ", ".join(sorted(st.session_state.uf_selected_set))
            if st.session_state.uf_selected_set else
            "All states"
        )
        st.markdown(f"#### Openings per year — {scope_lbl}")
        st.caption("This chart uses its **own** date range (does not affect the map or other charts).")

    with rc:
        # Local date filter defaults to the last 10 years (bounded at year 2000)
        today = date.today()
        default_start_local = today.replace(year=max(today.year - 10, 2000))
        local_rng = st.date_input(
            "Local date range (yearly openings)",
            (default_start_local, today),
            key="openings_local_range_geo",
            help="Affects only the yearly openings chart to the left."
        )

    # Robustly unpack the date_input result (tuple[date, date])
    if isinstance(local_rng, tuple) and len(local_rng) == 2:
        o_start, o_end = local_rng
        if o_start and o_end and o_start > o_end:
            o_start, o_end = o_end, o_start
    else:
        o_start, o_end = default_start_local, today

    o_start_str = o_start.strftime("%Y-%m-%d")
    o_end_str   = o_end.strftime("%Y-%m-%d")

    # Query and render the yearly openings area chart for the current UF scope (multi-select supported)
    openings = df_openings_by_year(
        fq_table,
        sorted(st.session_state.uf_selected_set) if st.session_state.uf_selected_set else None,
        start=o_start_str,
        end=o_end_str
    )

    if openings is not None and not openings.empty:
        st.plotly_chart(
            openings_area(openings, height=420),
            use_container_width=True,
            config={"displayModeBar": False}
        )
    else:
        st.info("No opening history found for the selected scope and local date range.")

# ───────────────────────── Seasonality (last 12 months) for a single UF 
with block():
    # This section shows a monthly series for the last 12 months, but only for a single UF.
    if not st.session_state.uf_selected_set:
        st.markdown("#### Monthly openings — last 12 months")
        st.caption("Select a single UF to visualize monthly openings in the last 12 months (seasonality).")
        st.info("Select one UF on the map or via the multiselect.")
    else:
        # If multiple UFs are selected, default to the first alphabetically for the seasonality lens.
        focus_uf = sorted(st.session_state.uf_selected_set)[0]
        st.markdown(f"#### Monthly openings — last 12 months (UF: {focus_uf})")
        st.caption("Shows new companies per month in the last 12 months for the selected UF — useful to read seasonality and short-term momentum.")

        last12 = df_openings_last_12mo(fq_table, focus_uf, end=end_str)
        if last12 is None or last12.empty:
            st.info("No monthly series available for the selected UF.")
        else:
            # plot_openings_monthly expects columns: 'mes' and 'openings'
            st.plotly_chart(
                plot_openings_monthly(last12),
                use_container_width=True,
                config={"displayModeBar": False}
            )


# ───────────────────────────── Regional analytics & indicators ─────────────────────────────
with block():
    st.markdown("### Regional analytics")

    # ——— Coastal vs. inland snapshot (stock) ————————————————————————————————
    uf_df2 = uf_df.copy()
    uf_df2["Coastal"] = uf_df2["uf"].isin(COASTAL_UF)
    coastal_total = int(uf_df2.loc[uf_df2["Coastal"], "qty"].sum())
    inland_total  = int(uf_df2.loc[~uf_df2["Coastal"], "qty"].sum())

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Coastal UFs — stock", fmt_num(coastal_total), sub="Sum of companies", variant=1)
    with c2:
        kpi_card("Inland UFs — stock", fmt_num(inland_total), sub="Sum of companies", variant=2)
    with c3:
        # Concentration index (share of top-5 cities inside the focused UF)
        if st.session_state.uf_selected_set:
            focus_uf = sorted(st.session_state.uf_selected_set)[0]
            conc = df_top5_concentration(fq_table, focus_uf, start=start_str, end=end_str)
            if conc:
                top5, tot = conc
                idx = (top5 / tot * 100.0) if tot else 0.0
                kpi_card("Top-5 city concentration", f"{idx:.1f}%", sub=f"UF {focus_uf}", variant=3)
            else:
                kpi_card("Top-5 city concentration", "—", sub="Data not available", variant=3)
        else:
            kpi_card("Top-5 city concentration", "—", sub="Select one UF", variant=3)

    # ——— North vs. Southeast gap — actives, MEI density, avg. capital —————————
if SHOW_REGION_GAP:
    with st.expander(
        "North vs. Southeast — gap: active rate, MEI density, average capital (winsorized)",
        expanded=True
    ):
        # --- Active rate -------------------------------------------------------
        act_sql = f"""
        WITH base AS (
            SELECT
                e.uf,
                CASE
                    WHEN COALESCE(LPAD(TRIM(CAST(e.situacao_cadastral AS TEXT)), 2, '0'), '') = '02'
                    THEN 1 ELSE 0
                END AS is_active
            FROM {fq_table} e
            WHERE {DATE_DT_EXPR} BETWEEN '{start_str}' AND '{end_str}'
        )
        SELECT uf, SUM(is_active)::bigint AS actives, COUNT(*)::bigint AS total
        FROM base
        GROUP BY uf;
        """
        act = _df_or_none(_load_df(act_sql))
        pctN = pctS = 0.0
        if act is not None and not act.empty:
            act["region"] = act["uf"].map(UF_TO_REGION)
            north = act[act["region"] == "N"]
            se    = act[act["region"] == "SE"]
            aN, tN = int(north["actives"].sum()), int(north["total"].sum())
            aS, tS = int(se["actives"].sum()),   int(se["total"].sum())
            pctN = (aN / tN * 100.0) if tN else 0.0
            pctS = (aS / tS * 100.0) if tS else 0.0

        # --- MEI density -------------------------------------------------------
        densN = densS = 0.0
        if SIMPLES_TBL:
            mei_sql = f"""
            WITH base AS (
                SELECT e.uf, e.cnpj_basico
                FROM {fq_table} e
                WHERE {DATE_DT_EXPR} BETWEEN '{start_str}' AND '{end_str}'
            ),
            du AS (
                SELECT DISTINCT uf, cnpj_basico FROM base
            ),
            m AS (
                SELECT du.uf, CASE WHEN s.opcao_mei = 'S' THEN 1 ELSE 0 END AS is_mei
                FROM du LEFT JOIN {SIMPLES_TBL} s USING (cnpj_basico)
            )
            SELECT uf, SUM(is_mei)::bigint AS mei, COUNT(*)::bigint AS total
            FROM m
            GROUP BY uf;
            """
            mei = _df_or_none(_load_df(mei_sql))
            if mei is not None and not mei.empty:
                mei["region"] = mei["uf"].map(UF_TO_REGION)
                mn = mei[mei["region"] == "N"]
                ms = mei[mei["region"] == "SE"]
                meiN, totN = int(mn["mei"].sum()),  int(mn["total"].sum())
                meiS, totS = int(ms["mei"].sum()),  int(ms["total"].sum())
                densN = (meiN / totN * 100.0) if totN else 0.0
                densS = (meiS / totS * 100.0) if totS else 0.0

        # --- Average capital (winsorized at P90 per UF) ------------------------
        cn = cs = 0.0
        if EMP_TBL:
            cap_sql = f"""
            WITH base AS (
                SELECT
                    e.uf,
                    NULLIF(REPLACE(emp.capital_social, ',', '.'), '')::numeric AS cap
                FROM {fq_table} e
                LEFT JOIN {EMP_TBL} emp USING (cnpj_basico)
                WHERE {DATE_DT_EXPR} BETWEEN '{start_str}' AND '{end_str}'
            ),
            p AS (
                SELECT uf, PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY cap) AS p90
                FROM base
                WHERE cap IS NOT NULL
                GROUP BY uf
            ),
            w AS (
                SELECT b.uf, LEAST(b.cap, p.p90) AS cap_w
                FROM base b
                JOIN p ON p.uf = b.uf
                WHERE b.cap IS NOT NULL
            )
            SELECT uf, AVG(cap_w) AS avg_cap_w
            FROM w
            GROUP BY uf;
            """
            cap = _df_or_none(_load_df(cap_sql))
            if cap is not None and not cap.empty:
                cap["region"] = cap["uf"].map(UF_TO_REGION)
                cn = float(cap.loc[cap["region"] == "N",  "avg_cap_w"].mean() or 0.0)
                cs = float(cap.loc[cap["region"] == "SE", "avg_cap_w"].mean() or 0.0)

        # ——— Three compact comparatives (North vs. Southeast) ————————————————
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Active rate — North", f"{pctN:.1f}%", delta=f"{(pctN - pctS):+.1f} vs SE")
        with c2:
            st.metric("MEI density — North", f"{densN:.1f}%", delta=f"{(densN - densS):+.1f} vs SE")
        with c3:
            st.metric("Avg. capital (P90-capped) — North", f"{cn:,.0f}", delta=f"{(cn - cs):+,.0f} vs SE")

hr()


# ───────────────────────────── Enterprise profile (by UF) ─────────────────────
