"""
dados_cnpj_baixa_dag.py
========================
Utility helpers used by *both* the notebook and the Airflow DAG that
handle the **download** phase of the “Dados Abertos CNPJ” workflow.

The module is 100 % self-contained – feel free to drop it under
`airflow/dags/utils/` (or anywhere on `PYTHONPATH`) and import:

    from utils.cnpj_download_helpers import (
        REPO_ROOT, BASE_DIR, DATA_DIR, ZIP_DIR,
        clean_folders, inventory_existing_zips,
        newest_month_links, download_zips, run_cnpj_download,
    )

Compatible with **plain scripts**, **Jupyter notebooks**, and
**Docker-based Airflow tasks**.

Author  : Vinicius Lamb
"""

# ---------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------
from __future__ import annotations

import logging
import math
import os
import sys
import time
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------
# Third-party – declared in requirements.txt / airflow image
# ---------------------------------------------------------------------
import requests
from bs4 import BeautifulSoup

# ═════════════════════════════════════════════════════════════════════
#  1) Path resolution – survives scripts, notebooks *and* containers
# ═════════════════════════════════════════════════════════════════════
def find_repo_root() -> Path:
    """
    Attempt to locate the repository root in a robust manner.

    1. If ``__file__`` is defined (running from a .py file) start there;
    2. Otherwise (interactive / notebook) start from ``Path.cwd()``;
    3. Walk upwards until a folder named **airflow** is found, which is
       our project convention for the repo root sentinel.
    """
    root = Path(__file__).resolve() if "__file__" in globals() else Path.cwd().resolve()
    while root != root.parent and not (root / "airflow").is_dir():
        root = root.parent
    return root


# ---- Canonical directories --------------------------------------------------
REPO_ROOT: Path = find_repo_root()
BASE_DIR: Path  = REPO_ROOT / "airflow" / "datasets"
DATA_DIR: Path  = BASE_DIR / "public-data"
ZIP_DIR: Path   = BASE_DIR / "public-zips"

# Ensure the directories exist both *outside* and *inside* the container
for folder in (DATA_DIR, ZIP_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# ---- Friendly logging -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s ➜ %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("cnpj-download")
log.info("📁  REPO_ROOT ➜ %s", REPO_ROOT)
log.info("📁  DATA_DIR  ➜ %s", DATA_DIR)
log.info("📁  ZIP_DIR   ➜ %s", ZIP_DIR)

# ═════════════════════════════════════════════════════════════════════
#  2) Folder-maintenance helpers
# ═════════════════════════════════════════════════════════════════════
def clean_folders() -> None:
    """
    **Delete** *every* previously-downloaded file inside ``DATA_DIR`` and
    ``ZIP_DIR``.  
    Use with care – there is no recycle bin!
    """
    for folder in (DATA_DIR, ZIP_DIR):
        for file_path in folder.glob("*"):
            try:
                file_path.unlink()
                print(f"🗑️  Deleted {file_path.name}")
            except PermissionError:
                print(f"⚠️  Permission denied for {file_path}")


def inventory_existing_zips() -> Dict[str, dt]:
    """
    Build (and pretty-print) an inventory of archives already present
    under ``ZIP_DIR``.

    Returns
    -------
    dict[str, datetime]
        Mapping **filename → last-modified timestamp**.

    When no files are found an empty dict is returned.
    """
    if not any(ZIP_DIR.iterdir()):
        print("ℹ️  No ZIP files found in", ZIP_DIR)
        return {}

    print("📦  ZIP files currently present:")
    inventory: Dict[str, dt] = {}

    for file_path in ZIP_DIR.iterdir():
        if file_path.is_dir():
            continue  # ignore directories

        modified = dt.fromtimestamp(file_path.stat().st_mtime)
        size_mb  = file_path.stat().st_size / 1_048_576
        inventory[file_path.name] = modified
        print(
            f"  • {file_path.name:<50} "
            f"{size_mb:6.1f} MB  –  {modified:%Y-%m-%d %H:%M}"
        )

    return inventory

# ═════════════════════════════════════════════════════════════════════
#  3) Scrape the Receita Federal index for the *latest* month
# ═════════════════════════════════════════════════════════════════════
RECEITA_ROOT: str         = "https://arquivos.receitafederal.gov.br/cnpj/dados_abertos_cnpj/"
HEADERS: dict[str, str]   = {"User-Agent": "Mozilla/5.0"}

def newest_month_links(max_links: int = 3) -> List[str]:
    """
    Discover the most-recent monthly folder (`YYYYMM/`) and return up to
    **`max_links`** direct `.zip` URLs.

    Parameters
    ----------
    max_links : int, default 3
        Hard cap – handy for unit tests and CI pipelines.

    Raises
    ------
    RuntimeError
        If the index cannot be reached or no ZIPs are found.

    Returns
    -------
    list[str]
        Fully-qualified HTTPS links ready to download.
    """
    # -- main index ------------------------------------------------------------
    print("🔎  Requesting main index …")
    try:
        index_resp = requests.get(RECEITA_ROOT, timeout=15, headers=HEADERS)
        index_resp.raise_for_status()
    except requests.RequestException as err:
        raise RuntimeError(f"Failed to fetch main index: {err}") from err

    soup_index = BeautifulSoup(index_resp.text, "lxml")

    # -- latest folder ---------------------------------------------------------
    month_dirs = sorted(a["href"] for a in soup_index.select('a[href^="20"]'))
    if not month_dirs:
        raise RuntimeError("Could not locate any monthly folders in index.")

    latest_folder = month_dirs[-1]        # e.g. "202503/"
    month_url     = RECEITA_ROOT + latest_folder
    print(f"  • Latest folder detected: {latest_folder}")

    # -- fetch that folder -----------------------------------------------------
    try:
        month_resp = requests.get(month_url, timeout=15, headers=HEADERS)
        month_resp.raise_for_status()
    except requests.RequestException as err:
        raise RuntimeError(f"Failed to open folder {latest_folder}: {err}") from err

    soup_month = BeautifulSoup(month_resp.text, "lxml")

    zip_urls = [
        href if href.startswith("http") else month_url + href
        for href in (a.get("href") for a in soup_month.find_all("a"))
        if href and href.lower().endswith(".zip")
    ]
    if not zip_urls:
        raise RuntimeError(f"No .zip files found under {latest_folder}")

    capped = zip_urls[:max_links]
    print(f"  • {len(zip_urls)} ZIPs available; returning the first {max_links}.")
    for i, url in enumerate(capped, 1):
        print(f"    - [{i}] {Path(url).name}")

    return capped

# ═════════════════════════════════════════════════════════════════════
#  4) Download ZIP archives (stream, skip duplicates)
# ═════════════════════════════════════════════════════════════════════
def download_zips(
    urls: List[str],
    destination: Path = ZIP_DIR,
    chunk_size: int = 1 << 20,  # 1 MiB
) -> None:
    """
    Stream-download every URL into *destination* (skipping existing files).

    Shows a progress line roughly every 5 %.

    Parameters
    ----------
    urls : list[str]
        HTTPS links to `.zip` files.
    destination : Path
        Folder where the archives will be stored.
    chunk_size : int
        Number of **bytes** per `iter_content` read; default 1 MiB.
    """
    if not urls:
        print("⚠️  No URLs supplied – nothing to download.")
        return

    destination.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        session.headers.update(HEADERS)

        total_files = len(urls)
        for idx, url in enumerate(urls, 1):
            fname  = Path(url).name
            local  = destination / fname

            # -- skip duplicates ------------------------------------------------
            if local.exists():
                print(f"✅ [{idx}/{total_files}] {fname} already present – skipping.")
                continue

            print(f"⬇️  [{idx}/{total_files}] Downloading {fname} …")
            start = time.time()

            try:
                with session.get(url, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    total_bytes   = int(resp.headers.get("content-length", 0))
                    bytes_written = 0
                    next_tick     = 5   # percentage for next progress print

                    with local.open("wb") as fp:
                        for chunk in resp.iter_content(chunk_size=chunk_size):
                            fp.write(chunk)
                            bytes_written += len(chunk)

                            # progress every ∼5 %
                            if total_bytes:
                                pct = bytes_written * 100 / total_bytes
                                if pct >= next_tick:
                                    mb = bytes_written / 1_048_576
                                    print(f"   • {pct:5.1f}% ({mb:,.1f} MB)")
                                    next_tick += 5

                dur = time.time() - start
                mb  = bytes_written / 1_048_576
                print(f"🎉  Finished {fname} in {dur:,.1f}s ({mb:,.1f} MB)\n")

            except requests.RequestException as err:
                print(f"❌  Error downloading {fname}: {err}")
                if local.exists():
                    local.unlink()   # remove partial file

    print("🏁  All downloads completed.")

# ═════════════════════════════════════════════════════════════════════
#  5) One-click runner – ties everything together
# ═════════════════════════════════════════════════════════════════════
def run_cnpj_download(limit: int = 3, clean_first: bool = False) -> None:
    """
    Convenience wrapper that orchestrates the **entire** workflow:

    1. Optionally wipe folders • ``clean_first=True``;
    2. Show inventory of ZIPs on disk (before);
    3. Scrape the latest month (returns up to ``limit`` URLs);
    4. Download any missing archives;
    5. Show final inventory + summary.

    Parameters
    ----------
    limit : int
        How many ZIPs to retrieve (quick tests = small number).
    clean_first : bool
        If *True* calls :pyfunc:`clean_folders` before doing anything else.
    """
    print("\n🚀  CNPJ end-to-end download started\n" + "-" * 60)

    # (1) Optional cleanup
    if clean_first:
        print("🧹  Cleaning target folders …")
        clean_folders()
        print("   Done.\n")

    # (2) Inventory before
    print("📑  Inventory *before* download:")
    pre_inv = inventory_existing_zips()
    print()

    # (3) Scrape latest folder
    try:
        links = newest_month_links(max_links=limit)
    except RuntimeError as err:
        print(f"❌  Aborting – {err}")
        return
    print()

    # (4) Download
    download_zips(links)
    print()

    # (5) Inventory after
    print("📑  Inventory *after* download:")
    post_inv = inventory_existing_zips()
    added = set(post_inv) - set(pre_inv)
    print(f"\n✅  Added {len(added)} new file(s): {', '.join(added) if added else '-'}")
    print("\n🏁  Workflow complete.")


# airflow/dags/cnpj_baixa_dag.py
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

default_args = dict(
    owner="Vinicius Lamb",
    start_date=days_ago(1),
    retry_delay=timedelta(seconds=30),
    max_active_runs=1,
    catchup=False
)

with DAG(
    dag_id="cnpj_download",
    schedule_interval="@monthly",
    catchup=False,
    default_args=default_args,
    tags=["cnpj", "download"],
) as dag:
    wipe = PythonOperator(task_id="clean_folders", python_callable=clean_folders)
    scrape = PythonOperator(
        task_id="fetch_links",
        python_callable=lambda **ctx: ctx["ti"].xcom_push(
            key="links", value=newest_month_links(max_links=3)
        ),
    )
    download = PythonOperator(
        task_id="download_zips",
        python_callable=lambda **ctx: download_zips(
            ctx["ti"].xcom_pull(key="links", task_ids="fetch_links")
        ),
    )

    wipe >> scrape >> download
