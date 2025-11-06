"""
postgre_connector.py
--------------------
Thin wrapper around *psycopg2* for projects that rely on a classic
PostgreSQL connection.

Environment variables expected (.env)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HOST_POSTG, PORT_POSTG, USER_POSTG, PASSWORD_POSTG, DATABASE_POSTG
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Sequence

import psycopg2
from psycopg2.extensions import connection as _PGConnection
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()  # read .env once at import time


class Postgres:
    """
    Minimal helper around *psycopg2*.

    * `connect_postgres()`  → opens one connection on demand.
    * `execute_sql()`       → generic DDL/DML runner (INSERT/UPDATE/DDL).
    * `collect_data()`      → SELECT into `list[dict]`.

    Example
    -------
    >>> pg = Postgres()
    >>> pg.execute_sql("CREATE TABLE foo(id INT);")
    >>> rows = pg.collect_data("SELECT 1 AS ok;")
    """

    # ------------------------------------------------------------------
    # 0)  Credentials from the environment
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self.host: str | None = os.getenv("HOST_POSTG")
        self.port: int = int(os.getenv("PORT_POSTG", 5432))
        self.user: str | None = os.getenv("USER_POSTG")
        self.password: str | None = os.getenv("PASSWORD_POSTG")
        self.database: str | None = os.getenv("DATABASE_POSTG")

    # ------------------------------------------------------------------
    # 1)  Raw connection (used by your loaders)
    # ------------------------------------------------------------------
    def connect_postgres(self) -> _PGConnection:
        """
        Open **one** psycopg2 connection using `.env` credentials.

        Returns
        -------
        psycopg2.extensions.connection
        """
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.database,
        )

    # ------------------------------------------------------------------
    # 2)  Generic runner for any SQL (DDL/DML)
    # ------------------------------------------------------------------
    def execute_sql(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        many: bool = False,
    ) -> None:
        """
        Run any SQL statement (CREATE, INSERT, UPDATE, …).

        Parameters
        ----------
        sql : str
            The statement or prepared query with `%s` placeholders.
        params : sequence, optional
            Parameters to replace the placeholders.
        many : bool, default **False**
            If **True** execute *executemany()* (requires `params`
            to be an iterable of iterables).

        The transaction is **always** committed; failures rollback
        and re-raise the exception.
        """
        conn: _PGConnection | None = None
        try:
            conn = self.connect_postgres()
            with conn.cursor() as cur:
                if many:
                    if params is None:
                        raise ValueError("`many=True` requires non-null `params`.")
                    cur.executemany(sql, params)   # type: ignore[arg-type]
                else:
                    cur.execute(sql, params)
            conn.commit()
        except Exception:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    # ------------------------------------------------------------------
    # 3)  SELECT helper → list[dict]
    # ------------------------------------------------------------------
    def collect_data(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run a SELECT and return the rows as `list[dict]`
        (column → value) using *RealDictCursor*.

        Returns
        -------
        list[dict]
            Empty list if the query yields no rows.
        """
        conn: _PGConnection | None = None
        try:
            conn = self.connect_postgres()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:  # type: ignore[arg-type]
                cur.execute(sql, params)
                return list(cur.fetchall())
        finally:
            if conn:
                conn.close()
