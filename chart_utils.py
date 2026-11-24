"""Shared Plotly layout helpers used across pages."""

import copy
import datetime
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from assets.styles import DEEP_GREEN, PLOTLY_TEMPLATE

# ---------------------------------------------------------------------------
# Export / display rules
#
# * The chart title is shown on the page only (as HTML above the chart). The
#   downloaded PNG contains just the chart; the title becomes the file name.
# * PNG export is rendered at a small fixed canvas (EXPORT_W x EXPORT_H CSS px,
#   x EXPORT_SCALE for resolution) so it drops into Word/PPT at ~1/8 A4 without
#   shrinking the text. Fonts are px-based, so a small canvas = relatively
#   large, readable text.
# * Legend sits above the plot area (outside it, auto-expanding the top
#   margin) and every axis uses automargin, so tick labels, axis titles and
#   legend entries never overlap each other or the data.
# ---------------------------------------------------------------------------
EXPORT_W, EXPORT_H, EXPORT_SCALE = 640, 400, 3
FONT_FAMILY = "Apple SD Gothic Neo, Noto Sans KR, Malgun Gothic, sans-serif"

PLOTLY_CONFIG = dict(
    toImageButtonOptions=dict(
        format="png", filename="chart",
        scale=EXPORT_SCALE, width=EXPORT_W, height=EXPORT_H,
    ),
    displayModeBar=True,
    displaylogo=False,
    modeBarButtonsToRemove=["select2d", "lasso2d"],
)

_TICK_COLOR   = "#1A1A1A"
_LABEL_COLOR  = "#1A1A1A"
_LEGEND_COLOR = "#1A1A1A"

_BAD_FN_CHARS = re.compile(r'[\:*?"<>|\r\n\t]+')


def safe_filename(title: str) -> str:
    """Turn a chart title (Korean or English) into a download-safe file name."""
    name = re.sub(r"<[^>]+>", "", title or "chart")          # strip HTML tags
    name = re.sub(r"\s*[|·]\s*", "_", name)                  # "a | b" -> "a_b"
    name = name.replace("/", "-")
    name = _BAD_FN_CHARS.sub("", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._ ")
    return (name or "chart")[:120]


def export_config(title: str, width: int | None = None, height: int | None = None,
                  scale: int | None = None) -> dict:
    cfg = copy.deepcopy(PLOTLY_CONFIG)
    opts = cfg["toImageButtonOptions"]
    opts["filename"] = safe_filename(title)
    if width:
        opts["width"] = width
    if height:
        opts["height"] = height
    if scale:
        opts["scale"] = scale
    return cfg


def chart_title(title: str, subtitle: str = "") -> None:
    sub = (f'<span style="font-weight:400;color:#757575;font-size:12px;margin-left:8px">'
           f'{subtitle}</span>') if subtitle else ""
    st.markdown(
        f'<div style="font-weight:700;color:{DEEP_GREEN};font-size:14px;margin:6px 0 0">'
        f'{title}{sub}</div>',
        unsafe_allow_html=True,
    )


def tidy_for_export(fig: go.Figure, legend: str = "top") -> go.Figure:
    """Strip the in-figure title and apply overlap-free spacing rules."""
    fig.update_layout(title=None)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    # rate axes: fixed 2 decimals (avoids ticks like 4.527% on secondary axes)
    fig.for_each_xaxis(lambda a: a.update(tickformat=".2f") if a.ticksuffix == "%" and not a.tickformat else None)
    fig.for_each_yaxis(lambda a: a.update(tickformat=".2f") if a.ticksuffix == "%" and not a.tickformat else None)
    if legend == "top":
        fig.update_layout(legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=11, color=_LEGEND_COLOR),
            bgcolor="rgba(255,255,255,0)", borderwidth=0,
            itemsizing="constant", tracegroupgap=0,
        ))
    elif legend == "none":
        fig.update_layout(showlegend=False)
    return fig


def show_chart(fig: go.Figure, title: str, *, subtitle: str = "",
               legend: str = "top", export_w: int | None = None,
               export_h: int | None = None, export_scale: int | None = None,
               fixed_size: bool = False, key: str | None = None) -> None:
    """Render a chart: title on the page (not in the PNG), PNG named after it.

    fixed_size=True keeps the figure's own width (heatmaps that need fixed
    cell space); otherwise the chart stretches to the container on screen and
    exports at the small default canvas.
    """
    chart_title(title, subtitle)
    tidy_for_export(fig, legend)
    cfg = export_config(title, export_w, export_h, export_scale)
    st.plotly_chart(fig, width="content" if fixed_size else "stretch",
                    config=cfg, key=key)


def base_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Shared look. `title` is ignored inside the figure (shown on page via
    show_chart instead); kept as a parameter for call-site compatibility."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        title=None,
        font=dict(family=FONT_FAMILY, size=12, color=_LEGEND_COLOR),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=11, color=_LEGEND_COLOR),
            bgcolor="rgba(255,255,255,0)", borderwidth=0,
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=False, showline=True, linecolor="#AAAAAA", linewidth=1.2,
        tickfont=dict(size=11, color=_TICK_COLOR),
        title_font=dict(size=11, color=_LABEL_COLOR),
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#E6E6E6", gridwidth=1,
        tickfont=dict(size=11, color=_TICK_COLOR),
        title_font=dict(size=11, color=_LABEL_COLOR),
        title_standoff=6,
        automargin=True,
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


_SECTOR_ORDER = ["국고채", "통안채", "공사/공단채", "은행채", "카드채", "기타금융채", "회사채"]
_RATING_ORDER = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"]
_NO_RATING = "무등급"


def _order(values, pref):
    return sorted(values, key=lambda v: (pref.index(v) if v in pref else len(pref), v))


def segment_picker(
    bond_df: pd.DataFrame,
    key: str,
    default_tenors: list[str] | None = None,
    tenor_options: list[str] | None = None,
    exclude_cats: tuple[str, ...] = (),
) -> list[tuple[str, str]]:
    """Pick (category, tenor) pairs with three rows of toggle pills —
    섹터 / 등급 / 만기 — instead of one multiselect per tenor.

    Only tenors that actually exist in the uploaded data are offered (or the
    given tenor_options, e.g. the base series' tenors), and only
    (category, tenor) combinations with data are returned. A collapsed
    "개별 계열 제외" box handles one-off exclusions.
    """
    from data.loader import TENOR_LABELS  # local import: avoid cycle at module load

    meta = bond_df[["category", "sector", "rating"]].drop_duplicates("category")
    meta = meta[~meta["category"].isin(exclude_cats)]
    have = set(map(tuple, bond_df[["category", "tenor"]].drop_duplicates().to_numpy()))

    sectors = _order(meta["sector"].unique().tolist(), _SECTOR_ORDER)
    ratings = _order([r or _NO_RATING for r in meta["rating"].unique()], _RATING_ORDER + [_NO_RATING])
    tenors = [t for t in TENOR_LABELS if t in set(bond_df["tenor"])]
    if tenor_options is not None:
        tenors = [t for t in tenors if t in tenor_options]
    d_ten = [t for t in (default_tenors or tenors) if t in tenors] or tenors[:1]

    with st.container(border=True):
        sel_sec = st.pills("섹터", sectors, selection_mode="multi", default=sectors, key=f"{key}_sec")
        sel_rat = st.pills("등급", ratings, selection_mode="multi", default=ratings, key=f"{key}_rat")
        sel_ten = st.pills("만기", tenors, selection_mode="multi", default=d_ten, key=f"{key}_ten")
        cand = meta[meta["sector"].isin(sel_sec or []) &
                    meta["rating"].replace("", _NO_RATING).isin(sel_rat or [])]
        cand_cats = _order(cand["category"].tolist(), [])
        with st.expander(f"개별 계열 제외 (현재 {len(cand_cats)}개 계열)", expanded=False):
            excl = st.multiselect("제외할 계열", cand_cats, default=[], key=f"{key}_excl",
                                  label_visibility="collapsed")

    tenor_idx = {t: i for i, t in enumerate(TENOR_LABELS)}
    return [(c, t) for c in cand_cats if c not in excl
            for t in sorted(sel_ten or [], key=tenor_idx.get) if (c, t) in have]
