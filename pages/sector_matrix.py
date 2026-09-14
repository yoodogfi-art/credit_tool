"""Page: Sector Matrix"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from assets.styles import DEEP_GREEN, HEATMAP_RATE, HEATMAP_DIVERG, PLOTLY_TEMPLATE
from data.loader import TENOR_LABELS
from chart_utils import PLOTLY_CONFIG, date_range_picker

_ALL_RATINGS = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-"]


def _init_state(df: pd.DataFrame) -> None:
    sectors = sorted(df["sector"].unique().tolist())
    if "mx_sectors" not in st.session_state:
        st.session_state["mx_sectors"] = sectors[:]
    else:
        existing = st.session_state["mx_sectors"]
        for s in sectors:
            if s not in existing:
                existing.append(s)
        st.session_state["mx_sectors"] = [s for s in existing if s in sectors]

    if "mx_ratings" not in st.session_state:
        st.session_state["mx_ratings"] = _ALL_RATINGS[:]


def _reorder_ui(label: str, state_key: str) -> list[str]:
    """Up/Down buttons to reorder + checkboxes to toggle visibility."""
    order = st.session_state[state_key]

    with st.expander(f"{label} 순서 / 표시", expanded=False):
        cols_h = st.columns([3, 1, 1, 2])
        cols_h[0].markdown("항목")
        cols_h[1].markdown("위")
        cols_h[2].markdown("아래")
        cols_h[3].markdown("표시")

        new_order = order[:]
        for i, item in enumerate(new_order):
            c0, c1, c2, c3 = st.columns([3, 1, 1, 2])
            c0.write(item)
            if i > 0 and c1.button("↑", key=f"{state_key}_up_{i}", use_container_width=True):
                new_order[i - 1], new_order[i] = new_order[i], new_order[i - 1]
                st.session_state[state_key] = new_order
                st.rerun()
            if i < len(new_order) - 1 and c2.button("↓", key=f"{state_key}_dn_{i}", use_container_width=True):
                new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
                st.session_state[state_key] = new_order
                st.rerun()
            c3.checkbox("", value=True, key=f"{state_key}_chk_{item}", label_visibility="collapsed")

        st.session_state[state_key] = new_order

    return [x for x in st.session_state[state_key]
            if st.session_state.get(f"{state_key}_chk_{x}", True)]


def render(df: pd.DataFrame) -> None:
    st.header("Sector Matrix")
    _init_state(df)

    st.markdown("**분석 기간**")
    d_start, d_end = date_range_picker(df, "mx")
    dff = df[df["date"].between(d_start, d_end)]
    if dff.empty:
        st.warning("선택한 기간에 데이터가 없습니다.")
        return

    all_cats     = sorted(dff["category"].unique().tolist())
    default_base = next((c for c in all_cats if "국고채" in c or "공사/공단채 AAA" in c), all_cats[0])

    c1, c2, c3 = st.columns([1, 2, 3])
    tenor    = c1.selectbox("기준 만기", TENOR_LABELS, index=TENOR_LABELS.index("3Y"), key="mx_tenor")
    mode     = c2.radio("표시 값", ["금리(%)", "스프레드(bp)"], horizontal=True, key="mx_mode")
    sp_base  = c3.selectbox("스프레드 기준",  all_cats,
                             index=all_cats.index(default_base) if default_base in all_cats else 0,
                             key="mx_base")

    col_s, col_r = st.columns(2)
    with col_s:
        sector_order = _reorder_ui("섹터", "mx_sectors")
    with col_r:
        rating_order = _reorder_ui("등급", "mx_ratings")

    if not sector_order or not rating_order:
        st.warning("섹터 또는 등급을 하나 이상 선택하세요.")
        return

    # Build matrix values
    base_yield: float | None = None
    if mode == "스프레드(bp)":
        s_base = dff[(dff["category"] == sp_base) & (dff["tenor"] == tenor)]
        base_yield = float(s_base.sort_values("date").iloc[-1]["yield"]) if not s_base.empty else None

    matrix: dict[tuple[str, str], float] = {}
    for cat in all_cats:
        sub = dff[dff["category"] == cat]
        if sub.empty:
            continue
        s = dff[(dff["category"] == cat) & (dff["tenor"] == tenor)]
        if s.empty:
            continue
        last = float(s.sort_values("date").iloc[-1]["yield"])
        if mode == "스프레드(bp)":
            val = round((last - base_yield) * 100, 1) if base_yield is not None else np.nan
        else:
            val = round(last, 3)
        matrix[(sub.iloc[0]["sector"], sub.iloc[0]["rating"])] = val

    suffix = "%" if mode == "금리(%)" else "bp"
    z, hover, text = [], [], []
    for sec in sector_order:
        rz, rh, rt = [], [], []
        for rat in rating_order:
            v = matrix.get((sec, rat), np.nan)
            ok = isinstance(v, (int, float)) and not np.isnan(v)
            rz.append(v if ok else np.nan)
            rh.append(f"{sec} {rat}: {v}{suffix}" if ok else f"{sec} {rat}: -")
            rt.append(f"{v:.2f}" if ok else "")
        z.append(rz); hover.append(rh); text.append(rt)

    cs  = HEATMAP_RATE if mode == "금리(%)" else HEATMAP_DIVERG
    fig = go.Figure(go.Heatmap(
        z=z, x=rating_order, y=sector_order,
        text=text, texttemplate="%{text}",
        hovertext=hover, hoverinfo="text",
        colorscale=cs, showscale=True,
        colorbar=dict(title=dict(text=suffix, side="right"), thickness=12, len=0.8),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=max(280, len(sector_order) * 44 + 80),
        title=dict(
            text=f"섹터 매트릭스  |  {tenor}  |  {mode}  |  {dff['date'].max().strftime('%Y-%m-%d')}",
            font=dict(color=DEEP_GREEN, size=13), x=0,
        ),
        font=dict(family="Apple SD Gothic Neo, Noto Sans KR, sans-serif", size=11),
        margin=dict(l=120, r=30, t=48, b=30),
        xaxis=dict(side="top"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # Data table
    rows = [
        {"섹터": sec, **{rat: (f"{matrix[(sec,rat)]:.2f}{suffix}" if (sec, rat) in matrix else "-")
                        for rat in rating_order}}
        for sec in sector_order
    ]
    st.dataframe(pd.DataFrame(rows).set_index("섹터"), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 카테고리 x 만기 히트맵")
    hm_cats = st.multiselect("계열", all_cats, default=all_cats[:8], key="hm_cats")
    hm_mode = st.radio("값", ["금리(%)", "1M 변화(bp)"], horizontal=True, key="hm_mode")

    if hm_cats:
        hm_z, hm_text = [], []
        for cat in hm_cats:
            rz, rt = [], []
            for tn in TENOR_LABELS:
                s = dff[(dff["category"] == cat) & (dff["tenor"] == tn)]
                if s.empty:
                    rz.append(np.nan); rt.append(""); continue
                if hm_mode == "금리(%)":
                    v = float(s.sort_values("date").iloc[-1]["yield"])
                    rz.append(v); rt.append(f"{v:.3f}%")
                else:
                    ys2 = s.set_index("date")["yield"].sort_index()
                    v   = (ys2.iloc[-1] - ys2.iloc[-22]) * 100 if len(ys2) >= 22 else np.nan
                    rz.append(v); rt.append(f"{v:.1f}bp" if not np.isnan(v) else "")
            hm_z.append(rz); hm_text.append(rt)

        cs2 = HEATMAP_RATE if hm_mode == "금리(%)" else HEATMAP_DIVERG
        fig2 = go.Figure(go.Heatmap(
            z=hm_z, x=TENOR_LABELS, y=hm_cats,
            text=hm_text, texttemplate="%{text}",
            hovertemplate="%{y} %{x}: %{text}<extra></extra>",
            colorscale=cs2, showscale=True,
        ))
        fig2.update_layout(
            template=PLOTLY_TEMPLATE,
            height=max(280, len(hm_cats) * 34 + 90),
            title=dict(text=f"카테고리 x 만기  |  {hm_mode}", font=dict(color=DEEP_GREEN, size=13), x=0),
            font=dict(family="Apple SD Gothic Neo, Noto Sans KR, sans-serif", size=10),
            margin=dict(l=190, r=30, t=48, b=30),
            xaxis=dict(side="top"),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)