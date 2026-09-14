"""Page: Signal Dashboard — base vs. every sector/rating spread heatmap."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from assets.styles import DEEP_GREEN, HEATMAP_DIVERG
from data.loader import TENOR_LABELS, POLICY_RATE_SECTOR
from chart_utils import PLOTLY_CONFIG, date_range_picker

_RATING_ORDER = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
    "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
    "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D",
]


def _rating_sort_key(r: str) -> tuple:
    try:
        return (0, _RATING_ORDER.index(r))
    except ValueError:
        return (1, r)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _series(df: pd.DataFrame, cat: str, tenor: str) -> pd.Series:
    s = df[(df["category"] == cat) & (df["tenor"] == tenor)]
    return s.set_index("date")["yield"].sort_index().dropna()


def _rolling_z(spread: pd.Series, window: int) -> pd.Series:
    min_periods = max(5, window // 3)
    mean = spread.rolling(window, min_periods=min_periods).mean()
    std = spread.rolling(window, min_periods=min_periods).std(ddof=0)
    z = (spread - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def _spread_and_z(base: pd.Series, cmp_series: pd.Series, window: int) -> tuple[float, float]:
    """Return (latest spread bp, latest rolling z) vs base."""
    idx = base.index.intersection(cmp_series.index)
    if len(idx) < 2:
        return np.nan, np.nan
    spread = ((cmp_series - base) * 100).reindex(idx).dropna()
    if spread.empty:
        return np.nan, np.nan
    z = _rolling_z(spread, window).dropna()
    return spread.iloc[-1], (z.iloc[-1] if len(z) else np.nan)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render(df: pd.DataFrame) -> None:
    st.header("Signal Dashboard")
    st.caption("Base 계열 대비 전체 섹터 x 등급 스프레드 / Z-score 히트맵")

    all_cats = sorted(df["category"].unique().tolist())
    default_base = next((c for c in all_cats if "국고채" in c), all_cats[0])

    st.markdown("**분석 기간**")
    d_start, d_end = date_range_picker(df, "sig")
    dff = df[df["date"].between(d_start, d_end)]
    if dff.empty:
        st.warning("선택한 기간에 데이터가 없습니다.")
        return

    c1, c2, c3, c4 = st.columns(4)
    tenor = c1.selectbox("만기", TENOR_LABELS,
                          index=TENOR_LABELS.index("3Y") if "3Y" in TENOR_LABELS else 0,
                          key="sig_tenor")
    base_cat = c2.selectbox("기준(Base) 계열", all_cats,
                             index=all_cats.index(default_base) if default_base in all_cats else 0,
                             key="sig_base_cat")
    z_window = c3.number_input("Z-score 산출기간 (영업일)", value=60, min_value=10, max_value=500, key="sig_zwin")
    z_thresh = c4.number_input("Z-score 임계값", value=2.0, min_value=0.5, max_value=5.0, step=0.1, key="sig_zthresh")

    # Only sectors that actually carry a rating can sit on the grid
    # (government/policy-rate style series have no rating and are excluded).
    rated_sectors = sorted(
        s for s in dff["sector"].unique()
        if s != POLICY_RATE_SECTOR and (dff.loc[dff["sector"] == s, "rating"] != "").any()
    )
    if not rated_sectors:
        st.warning("등급 구분이 있는 계열이 없습니다.")
        return

    sel_sectors = st.multiselect(
        "표시할 섹터 (해제하면 히트맵에서 제외)", rated_sectors,
        default=rated_sectors, key="sig_sectors",
    )
    if not sel_sectors:
        st.info("표시할 섹터를 하나 이상 선택하세요.")
        return

    ratings_present = sorted(
        {r for r in dff.loc[dff["sector"].isin(sel_sectors), "rating"].unique() if r},
        key=_rating_sort_key,
    )
    if not ratings_present:
        st.warning("선택한 섹터에 등급 데이터가 없습니다.")
        return

    base_series = _series(dff, base_cat, tenor)
    if base_series.empty:
        st.warning(f"Base 계열의 {tenor} 데이터가 없습니다.")
        return

    cats_present = set(dff["category"].unique())
    z_grid, sp_grid, text_z, text_sp, hover = [], [], [], [], []
    breach_n, total_n = 0, 0
    for sec in sel_sectors:
        z_row, sp_row, tz_row, ts_row, hv_row = [], [], [], [], []
        for rat in ratings_present:
            cat = f"{sec} {rat}".strip()
            if cat == base_cat or cat not in cats_present:
                z_row.append(np.nan); sp_row.append(np.nan)
                tz_row.append(""); ts_row.append("")
                hv_row.append(f"{sec} {rat}: 데이터 없음")
                continue
            cmp_series = _series(dff, cat, tenor)
            sp_now, z_now = _spread_and_z(base_series, cmp_series, int(z_window))
            z_row.append(z_now); sp_row.append(sp_now)
            tz_row.append(f"{z_now:+.2f}" if not np.isnan(z_now) else "")
            ts_row.append(f"{sp_now:+.0f}" if not np.isnan(sp_now) else "")
            hv_row.append(
                f"{cat} vs {base_cat}<br>스프레드: "
                f"{sp_now:+.1f}bp<br>Z-score: {z_now:+.2f}"
                if not np.isnan(sp_now) else f"{cat}: 데이터 없음"
            )
            if not np.isnan(z_now):
                total_n += 1
                if abs(z_now) >= z_thresh:
                    breach_n += 1
        z_grid.append(z_row); sp_grid.append(sp_row)
        text_z.append(tz_row); text_sp.append(ts_row)
        hover.append(hv_row)

    st.markdown(
        f'<div style="background:#F7F8F5;border-radius:6px;padding:14px 18px;'
        f'border-left:4px solid {DEEP_GREEN};margin:12px 0">'
        f'<div style="font-size:11px;color:#888;margin-bottom:4px">종합</div>'
        f'<div style="font-size:14px;font-weight:600;color:{DEEP_GREEN}">'
        f'{total_n}개 계열 중 {breach_n}개 임계값(±{z_thresh}) 초과 &nbsp;|&nbsp; '
        f'기준: {base_cat} {tenor}</div></div>',
        unsafe_allow_html=True,
    )

    def _heatmap(z: list, text: list, hover_text: list, suffix: str, zmid: float | None) -> go.Figure:
        fig = go.Figure(go.Heatmap(
            z=z, x=ratings_present, y=sel_sectors,
            text=text, texttemplate="%{text}",
            hovertext=hover_text, hoverinfo="text",
            colorscale=HEATMAP_DIVERG, zmid=zmid, showscale=True,
            colorbar=dict(title=dict(text=suffix, side="right"), thickness=12, len=0.8),
            textfont=dict(size=11),
        ))
        fig.update_layout(
            height=max(260, len(sel_sectors) * 42 + 60),
            margin=dict(l=110, r=30, t=10, b=30),
            font=dict(family="Apple SD Gothic Neo, Noto Sans KR, sans-serif", size=11),
            xaxis=dict(side="top"),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        return fig

    st.markdown("#### Z-score 히트맵")
    fig_z = _heatmap(z_grid, text_z, hover, "Z", zmid=0)
    st.plotly_chart(fig_z, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("#### 스프레드 히트맵 (bp)")
    fig_sp = _heatmap(sp_grid, text_sp, hover, "bp", zmid=None)
    st.plotly_chart(fig_sp, use_container_width=True, config=PLOTLY_CONFIG)
