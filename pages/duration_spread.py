"""Page: Duration Spread — spread per year of (Macaulay) duration.

Duration is approximated for a plain-vanilla bond that pays a coupon every
6 months and is priced at par (coupon rate = its own yield to maturity).
No other complexity (calls, floaters, amortization, day-count conventions,
...) is modeled — this is a rough, consistent yardstick for comparing
spread "richness" across maturities and credits, not a pricing tool.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from assets.styles import DEEP_GREEN, HEATMAP_DIVERG
from data.loader import TENOR_LABELS, POLICY_RATE_SECTOR
from chart_utils import PLOTLY_CONFIG, date_range_picker

TENOR_YEARS = dict(zip(TENOR_LABELS, [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5]))

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
# Duration
# ---------------------------------------------------------------------------
def macaulay_duration_years(tenor_years: float, yield_pct: float) -> float:
    """Macaulay duration (years) of a par-priced, semiannual-coupon bond."""
    if tenor_years is None or np.isnan(tenor_years) or tenor_years <= 0 or np.isnan(yield_pct):
        return np.nan
    m = 2  # semiannual coupons
    n = max(1, round(tenor_years * m))
    y = yield_pct / 100.0
    if y <= 0:
        return n / m  # degenerate case: fall back to time-weighted average maturity
    per_rate = y / m
    coupon = per_rate * 100.0
    periods = np.arange(1, n + 1)
    cash_flows = np.full(n, coupon, dtype=float)
    cash_flows[-1] += 100.0
    disc = (1 + per_rate) ** periods
    pv = cash_flows / disc
    price = pv.sum()
    mac_dur_periods = (periods * pv).sum() / price
    return mac_dur_periods / m


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _latest_yield(dff: pd.DataFrame, cat: str, tenor: str) -> float:
    s = dff[(dff["category"] == cat) & (dff["tenor"] == tenor)].sort_values("date")
    return float(s.iloc[-1]["yield"]) if not s.empty else np.nan


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render(df: pd.DataFrame) -> None:
    st.header("Duration Spread")
    st.caption("스프레드 ÷ 조정 듀레이션(6개월 이표, 액면발행 가정) — 만기별 스프레드의 '위험 대비 캐리'를 비교합니다.")

    st.markdown("**분석 기간**")
    d_start, d_end = date_range_picker(df, "dur")
    dff = df[df["date"].between(d_start, d_end)]
    if dff.empty:
        st.warning("선택한 기간에 데이터가 없습니다.")
        return

    all_cats = sorted(dff["category"].unique().tolist())
    default_base = next((c for c in all_cats if "국고채" in c), all_cats[0])

    c1, c2 = st.columns([2, 3])
    base_cat = c1.selectbox("기준(Base) 계열", all_cats,
                             index=all_cats.index(default_base) if default_base in all_cats else 0,
                             key="dur_base_cat")

    all_sectors = sorted(s for s in dff["sector"].unique() if s != POLICY_RATE_SECTOR)
    sel_sectors = c2.multiselect("표시할 섹터", all_sectors, default=all_sectors, key="dur_sectors")
    if not sel_sectors:
        st.info("표시할 섹터를 하나 이상 선택하세요.")
        return

    cat_df = dff[dff["sector"].isin(sel_sectors)][["category", "sector", "rating"]].drop_duplicates()
    cat_df = cat_df[cat_df["category"] != base_cat]
    if cat_df.empty:
        st.warning("표시할 계열이 없습니다.")
        return

    sector_rank = {s: i for i, s in enumerate(sel_sectors)}
    cat_info = cat_df.set_index("category")

    def _entity_key(cat: str) -> tuple:
        info = cat_info.loc[cat]
        return (sector_rank.get(info["sector"], 999), _rating_sort_key(info["rating"]))

    entities = sorted(cat_info.index.unique().tolist(), key=_entity_key)

    z_mat = np.full((len(entities), len(TENOR_LABELS)), np.nan)
    text = [["" for _ in TENOR_LABELS] for _ in entities]
    hover = [["" for _ in TENOR_LABELS] for _ in entities]

    for i, cat in enumerate(entities):
        for j, tenor in enumerate(TENOR_LABELS):
            ent_y = _latest_yield(dff, cat, tenor)
            base_y = _latest_yield(dff, base_cat, tenor)
            if np.isnan(ent_y) or np.isnan(base_y):
                hover[i][j] = f"{cat} {tenor}: 데이터 없음"
                continue
            sp_bp = (ent_y - base_y) * 100
            dur = macaulay_duration_years(TENOR_YEARS[tenor], ent_y)
            if not dur or dur <= 0:
                hover[i][j] = f"{cat} {tenor}: 듀레이션 계산 불가"
                continue
            ratio = sp_bp / dur
            z_mat[i, j] = ratio
            text[i][j] = f"{ratio:+.0f}"
            hover[i][j] = (
                f"{cat} {tenor} vs {base_cat}<br>"
                f"스프레드: {sp_bp:+.1f}bp<br>듀레이션: {dur:.2f}y<br>"
                f"스프레드/듀레이션: {ratio:+.1f}bp/y"
            )

    st.markdown(
        f'<div style="background:#F7F8F5;border-radius:6px;padding:14px 18px;'
        f'border-left:4px solid {DEEP_GREEN};margin:12px 0">'
        f'<div style="font-size:11px;color:#888;margin-bottom:4px">종합</div>'
        f'<div style="font-size:14px;font-weight:600;color:{DEEP_GREEN}">'
        f'기준: {base_cat} &nbsp;|&nbsp; {len(entities)}개 계열 x {len(TENOR_LABELS)}개 만기</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("셀 = (스프레드 vs 기준, bp) ÷ (듀레이션, 년). 값이 클수록 위험(듀레이션) 대비 캐리가 두텁다는 의미입니다.")

    fig = go.Figure(go.Heatmap(
        z=z_mat.tolist(), x=TENOR_LABELS, y=entities,
        text=text, texttemplate="%{text}",
        hovertext=hover, hoverinfo="text",
        colorscale=HEATMAP_DIVERG, zmid=0, showscale=True,
        colorbar=dict(title=dict(text="bp/y", side="right"), thickness=12, len=0.8),
        textfont=dict(size=10),
    ))
    fig.update_layout(
        height=max(320, len(entities) * 34 + 60),
        margin=dict(l=140, r=30, t=10, b=10),
        font=dict(family="Apple SD Gothic Neo, Noto Sans KR, sans-serif", size=11),
        xaxis=dict(side="top"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
