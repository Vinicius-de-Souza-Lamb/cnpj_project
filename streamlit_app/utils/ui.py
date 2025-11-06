# utils/ui.py — shared UI building blocks for Streamlit dashboards
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Tuple
from string import Template

import streamlit as st

__all__ = [
    "THEME",
    "inject_theme",
    "fmt_num",
    "fmt_money",
    "kpi_card",
    "stat_tile",
    "block",
    "map_frame",
    "date_filter_compact",
    "hr",
]

# -------------------------------------------------------------------
# THEME — single source of truth for colors and surfaces
# Keep it lean: only tokens that are reused across pages.
# -------------------------------------------------------------------
THEME = {
    "bg_top": "#0b1022",
    "bg_bottom": "#101733",
    "text": "#E7ECF6",
    "muted": "#A9B3C8",
    "ring": "rgba(255,255,255,0.16)",
    "card": "rgba(255,255,255,0.06)",

    # Hero gradients (Material-ish)
    "hero1_a": "#0B57D0", "hero1_b": "#4A8EFF",
    "hero2_a": "#018786", "hero2_b": "#20C997",
    "hero3_a": "#FFB500", "hero3_b": "#FFD55A",

    # Accent palette for small tiles / charts
    "aqua": "#17B890",
    "red": "#FF6B6B",
    "violet": "#8E7DFF",

    # Heat scale for choropleth
    "heat0": "#0d1224", "heat1": "#172a4a", "heat2": "#214c80", "heat3": "#3f8cff", "heat4": "#9fd0ff",
}


# -------------------------------------------------------------------
# GLOBAL CSS — injected once per page. Uses Template to avoid f-string
# brace escaping headaches when writing CSS blocks.
# -------------------------------------------------------------------
def inject_theme(theme: dict = THEME) -> None:
    css_tpl = Template(
        """
        <style>
        :root{
          --text: $text; --muted: $muted; --ring: $ring; --card: $card;
        }
        .stApp{
          color:var(--text);
          background:
            radial-gradient(1200px 600px at 10% -10%, #15224b 0%, transparent 40%),
            linear-gradient(180deg, $bg_top 0%, $bg_bottom 100%);
        }
        [data-testid="stSidebar"] > div:first-child{
          backdrop-filter: blur(10px) saturate(160%);
          background: rgba(255,255,255,.06);
          border-right: 1px solid var(--ring);
        }

        /* Generic glass block */
        .block{
          background: var(--card);
          border: 1px solid var(--ring);
          border-radius: 18px;
          padding: 18px;
          backdrop-filter: blur(10px);
          box-shadow: 0 20px 40px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.04);
        }

        /* Divider */
        .hr{ height:1px; width:100%;
          background: linear-gradient(90deg, transparent, rgba(255,255,255,.12), transparent);
          margin:8px 0 18px 0; }

        /* KPI — liquid glass */
        .kpi{
          position:relative; overflow:hidden; border-radius:16px; padding:16px 16px 14px 16px;
          border:1px solid var(--ring); color:#fff; min-height:140px;
          background: linear-gradient(180deg, rgba(255,255,255,.10), rgba(255,255,255,.04));
          box-shadow: 0 24px 40px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.05);
          transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }
        .kpi:hover{ transform: translateY(-3px); box-shadow:0 18px 32px rgba(0,0,0,.35); border-color: rgba(255,255,255,.28); }
        .kpi .label{ font-size:13px; font-weight:700; letter-spacing:.5px; text-transform:uppercase; opacity:.95; }
        .kpi .value{ font-size:38px; font-weight:900; line-height:1; margin-top:6px; }
        .kpi .sub{ color: rgba(255,255,255,.92); font-size:12px; margin-top:2px; }
        .kpi .tag{
          position:absolute; right:10px; top:10px; font-size:11px; font-weight:700;
          padding:4px 8px; border-radius:999px; background: rgba(93,140,255,.18); color:#cfe0ff; border:1px solid rgba(93,140,255,.35);
        }
        /* subtle colored accents for numbers */
        .value .accent{ color:#9ED8B7; text-shadow: 0 0 10px rgba(34,176,125,.35); }
        .value .accent-b{ color:#AFC6FF; text-shadow: 0 0 10px rgba(93,140,255,.35); }

        /* Small stat tile */
        .tile{
          border-radius:16px; padding:16px; border:1px solid var(--ring);
          transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }
        .tile:hover{ transform: translateY(-2px); box-shadow: 0 12px 22px rgba(0,0,0,.28); border-color: rgba(255,255,255,.22); }
        .tile .label{ font-size:12px; font-weight:700; letter-spacing:.5px; text-transform:uppercase; opacity:.9; }
        .tile .value{ font-size:32px; font-weight:900; line-height:1; margin-top:4px; }

        /* Map frame (glass + vignette) */
        .map-frame{
          position:relative; border-radius:20px; overflow:hidden; padding:12px;
          background:
            radial-gradient(800px 300px at 10% -20%, rgba(93,140,255,.15), transparent 50%),
            radial-gradient(600px 300px at 120% 120%, rgba(34,176,125,.12), transparent 50%),
            linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
          border: 1px solid rgba(255,255,255,.18);
          box-shadow: 0 30px 60px rgba(0,0,0,.45), inset 0 0 0 1px rgba(255,255,255,.06), 0 0 0 2px rgba(93,140,255,.25);
        }
        .map-vignette::after{ content:""; position:absolute; inset:0; pointer-events:none; border-radius:18px; box-shadow: inset 0 0 80px rgba(0,0,0,.45); }
        .legend-mini{
          position:absolute; bottom:12px; right:14px; z-index:10;
          padding:6px 8px; border-radius:12px; border:1px solid rgba(255,255,255,.22);
          background: rgba(0,0,0,.45); backdrop-filter: blur(6px); font-size:11px;
        }
        .legend-bar{
          width:120px; height:8px; border-radius:6px; margin-top:4px;
          background: linear-gradient(90deg, $heat0, $heat1, $heat2, $heat3, $heat4);
        }

        /* Compact date input (used by date_filter_compact) */
        .date-compact .stDateInput > div > div{
          background: rgba(255,255,255,.06);
          border: 1px solid var(--ring);
          border-radius: 12px;
          padding: 2px 6px;
        }
        </style>
        """
    )
    st.markdown(
        css_tpl.substitute(
            text=theme["text"],
            muted=theme["muted"],
            ring=theme["ring"],
            card=theme["card"],
            bg_top=theme["bg_top"],
            bg_bottom=theme["bg_bottom"],
            heat0=theme["heat0"],
            heat1=theme["heat1"],
            heat2=theme["heat2"],
            heat3=theme["heat3"],
            heat4=theme["heat4"],
        ),
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# FORMATTERS — locale-friendly helpers (pt-BR style)
# -------------------------------------------------------------------
def fmt_num(x: int | float | None) -> str:
    if x is None:
        return "—"
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return "—"


def fmt_money(x: int | float | None) -> str:
    if x is None:
        return "—"
    return f"R$ {float(x):,.2f}".replace(",", " ").replace(".", ",")


# -------------------------------------------------------------------
# UI BRICKS — composable building blocks
# -------------------------------------------------------------------
def kpi_card(label: str, value: str, *, sub: str | None = None, variant: int = 1) -> None:
    """
    Liquid-glass hero KPI. `variant` controls the gradient (1..3).
    The value supports inline spans like <span class="accent">...</span>.
    """
    v = max(1, min(3, variant))
    a = THEME[f"hero1_a" if v == 1 else f"hero{v}_a"]
    b = THEME[f"hero1_b" if v == 1 else f"hero{v}_b"]
    st.markdown(
        f"""
        <div class="kpi" style="background:linear-gradient(135deg,{a} 0%,{b} 100%);">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          {f'<div class="sub">{sub}</div>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_tile(label: str, value: str, *, color: str) -> None:
    """
    Small statistic tile with a single accent color.
    `color` should be a solid hex; a subtle gradient is applied automatically.
    """
    st.markdown(
        f"""
        <div class="tile" style="background: linear-gradient(135deg,{color} 0%, {color}99 100%); color:#0b0e17;">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def block():
    """
    Glass "block" container. Usage:
        with block():
    """
    st.markdown('<div class="block">', unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


def hr():
    """A thin, elegant divider that matches the theme."""
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


def map_frame(legend_label: str = "Density"):
    """
    Wrap a Plotly map inside a styled glass frame with a built-in legend.
    Usage:
        with map_frame("Legend text"):
            st.plotly_chart(fig, use_container_width=True)
    """
    @contextmanager
    def _frame():
        st.markdown('<div class="map-frame map-vignette">', unsafe_allow_html=True)
        try:
            yield
        finally:
            st.markdown(
                f'<div class="legend-mini">{legend_label}<div class="legend-bar"></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    return _frame()


# -------------------------------------------------------------------
# DATE FILTER — compact range selector (no preset buttons)
# Returns ('YYYY-MM-DD', 'YYYY-MM-DD')
# -------------------------------------------------------------------
def date_filter_compact(default_months: int = 12) -> Tuple[str, str]:
    """Compact calendar range picker suitable for placing above charts."""
    today = datetime.now().date()
    start_default = today - timedelta(days=30 * default_months)
    st.markdown('<div class="date-compact">', unsafe_allow_html=True)
    rng = st.date_input(
        "Date range",
        (start_default, today),
        label_visibility="collapsed",
        help="Applied to all metrics.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if isinstance(rng, tuple) and len(rng) == 2:
        start, end = rng
    else:
        start, end = start_default, today
    if start > end:
        start, end = end, start
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
