"""Shared Plotly layout helpers used across pages."""

import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from assets.styles import DEEP_GREEN, PLOTLY_TEMPLATE

# ---------------------------------------------------------------------------
# Export config — sized for 8 charts per A4 page
#
# Strategy: small base dimensions (640×390) so the chart occupies roughly
# 1/8 of an A4 when inserted into Word/PPT at its natural size.
# scale=3 gives 1920×1170 px — enough resolution for crisp print without
# the fonts becoming illegible at the final small print size.
#
# Font sizes are deliberately large relative to the canvas so they stay
# readable when the image is placed at ~95×68 mm on paper.
# ---------------------------------------------------------------------------
PLOTLY_CONFIG = dict(
    toImageButtonOptions=dict(
        format="png",
        filename="chart",
        scale=3,          # 640*3 = 1920 px wide — sharp at small print size
        width=640,        # ~95 mm at 170 dpi; fits 2-up on A4
        height=390,       # ~68 mm; fits 4-down on A4  (2×4 = 8 per page)
    ),
    displayModeBar=True,
    modeBarButtonsToRemove=["select2d", "lasso2d"],
)

# Text colors — use near-black so nothing fades at small size
_TICK_COLOR   = "#1A1A1A"   # axis tick labels
_LABEL_COLOR  = "#1A1A1A"   # axis titles
_TITLE_COLOR  = DEEP_GREEN  # chart title (already dark green)
_LEGEND_COLOR = "#1A1A1A"


def base_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(color=_TITLE_COLOR, size=14, family="Apple SD Gothic Neo, Noto Sans KR, sans-serif"),
            x=0,
        ),
        font=dict(
            family="Apple SD Gothic Neo, Noto Sans KR, sans-serif",
            size=13,        # base font — governs legend & hover text
            color=_LEGEND_COLOR,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color=_LEGEND_COLOR),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="#C8C8C8",
            borderwidth=1,
        ),
        # Tighter margins so the data area takes more of the small canvas
        margin=dict(l=58, r=64, t=46, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor="#AAAAAA",
        linewidth=1.5,
        tickfont=dict(size=12, color=_TICK_COLOR),
        title_font=dict(size=12, color=_LABEL_COLOR),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E2E2E2",
        gridwidth=1,
        tickfont=dict(size=12, color=_TICK_COLOR),
        title_font=dict(size=12, color=_LABEL_COLOR),
    )
    return fig


def date_range_picker(
    df: pd.DataFrame,
    key: str,
    default_days: int = 365,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    min_d = df["date"].min().date()
    max_d = df["date"].max().date()

    # Read from the sidebar widgets (keys sb_date_start / sb_date_end set in app.py)
    g_start = st.session_state.get("sb_date_start")
    g_end   = st.session_state.get("sb_date_end")

    default_start = g_start if g_start else max(max_d - datetime.timedelta(days=default_days), min_d)
    default_end   = g_end   if g_end   else max_d

    c1, c2 = st.columns(2)
    with c1:
        start_d = st.date_input(
            "시작일", value=default_start,
            min_value=min_d, max_value=max_d,
            key=f"{key}_start",
        )
    with c2:
        end_d = st.date_input(
            "종료일", value=default_end,
            min_value=min_d, max_value=max_d,
            key=f"{key}_end",
        )
    if start_d > end_d:
        start_d, end_d = end_d, start_d
    return pd.Timestamp(start_d), pd.Timestamp(end_d)


def category_tenor_picker(
    categories: list[str],
    tenors: list[str],
    key: str,
    default: str = "single",
    default_tenor: str | None = None,
) -> list[tuple[str, str]]:
    """One "계열"-style multiselect per maturity, laid out in a grid so
    every maturity is its own clearly-labeled, always-visible control (a
    single flat "category maturity" multiselect made it easy to lose track
    of which maturities were actually selected). Options are individual
    categories — sector + rating, e.g. "공사채 AAA", "공사채 AA-" — not bare
    sector names, so specific ratings can be included or excluded directly.
    Each multiselect already has its own select-all and per-item clear
    ("x") affordance built in, so no extra buttons are added here.

    default="all" preselects every category for every maturity; default="single"
    preselects every category only for default_tenor (falls back to the
    first tenor), leaving other maturities empty. Returns the selected
    (category, tenor) pairs.
    """
    default_tenor = default_tenor if default_tenor in tenors else (tenors[0] if tenors else None)

    with st.expander("계열 x 만기 선택 (만기별로 표시할 계열을 선택하세요)", expanded=True):
        pairs: list[tuple[str, str]] = []
        per_row = 5
        for row_start in range(0, len(tenors), per_row):
            row_tenors = tenors[row_start:row_start + per_row]
            cols = st.columns(len(row_tenors))
            for col, tenor in zip(cols, row_tenors):
                if default == "all" or (default == "single" and tenor == default_tenor):
                    widget_default = list(categories)
                else:
                    widget_default = []
                with col:
                    sel = st.multiselect(tenor, categories, default=widget_default, key=f"{key}_tenor_{tenor}")
                pairs.extend((c, tenor) for c in sel)

    return pairs