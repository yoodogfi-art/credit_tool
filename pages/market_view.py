"""Page: Market View"""

import copy
import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from assets.styles import CHART_COLORS, CORAL, DEEP_GREEN, OLIVE, POLICY_COLOR, POLICY_FILL_COLOR
from data.loader import TENOR_LABELS, get_bond_data, get_policy_rate
from chart_utils import PLOTLY_CONFIG, base_layout, date_range_picker

_TENOR_ORDER = {t: i for i, t in enumerate(TENOR_LABELS)}
_GREEN_SHADES = ["#2D3F38", "#4A5E35", "#4E9B5A", "#8DC175", "#8DD5C8", "#DDE8C0",
                 "#8DB8A5", "#9A7085", "#8A9E96", "#B0BDB4"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _bond_cats(df: pd.DataFrame) -> list[str]:
    return sorted(get_bond_data(df)["category"].unique().tolist())


def _policy_cats(df: pd.DataFrame) -> list[str]:
    pr = get_policy_rate(df)
    return sorted(pr["category"].unique().tolist()) if not pr.empty else []


def _add_policy_traces(fig: go.Figure, pr_df: pd.DataFrame,
                       d_start: pd.Timestamp, d_end: pd.Timestamp,
                       secondary_y: bool = False) -> None:
    for cat in sorted(pr_df["category"].unique()):
        s = pr_df[(pr_df["category"] == cat) &
                  pr_df["date"].between(d_start, d_end)].sort_values("date")
        if s.empty:
            continue
        country = s.iloc[0]["rating"]
        fig.add_trace(
            go.Scatter(
                x=s["date"], y=s["yield"],
                name=f"{country} 기준금리",
                line=dict(color=POLICY_COLOR, width=2, dash="dashdot"),
                line_shape="hv",
                hovertemplate=f"{country} 기준금리: %{{y:.2f}}%<extra></extra>",
            ),
            secondary_y=secondary_y,
        )


def _policy_banner(df: pd.DataFrame) -> None:
    pr_df = get_policy_rate(df)
    if pr_df.empty:
        return
    cols = st.columns(pr_df["category"].nunique())
    for col, cat in zip(cols, sorted(pr_df["category"].unique())):
        s    = pr_df[pr_df["category"] == cat].sort_values("date")
        cur  = s.iloc[-1]["yield"]
        prev = s.iloc[-2]["yield"] if len(s) > 1 else cur
        chg  = cur - prev
        dir_str = f"+{chg:.2f}%" if chg > 0 else f"{chg:.2f}%"
        col.metric(
            label=f"{s.iloc[-1]['rating']} 기준금리",
            value=f"{cur:.2f}%",
            delta=dir_str,
        )


# ---------------------------------------------------------------------------
# Tab 1: Rate & Spread Summary Table
# ---------------------------------------------------------------------------
def _tab_summary(df: pd.DataFrame) -> None:
    bond_df  = get_bond_data(df)
    all_cats = _bond_cats(df)
    tenors   = ["6M", "1Y", "2Y", "3Y", "5Y"]

    _policy_banner(df)
    st.markdown("")

    c1, c2 = st.columns([4, 2])
    sel_cats = c1.multiselect(
        "표시 계열", all_cats,
        default=[c for c in all_cats if any(x in c for x in
            ["공사/공단채 AAA", "은행채 AAA", "카드채 AA", "회사채 AA-", "회사채 AA"])][:6],
        key="tbl_cats",
    )
    ref_cat = c2.selectbox(
        "스프레드 기준", all_cats,
        index=next((i for i, c in enumerate(all_cats) if "국고채" in c or "공사/공단채 AAA" in c), 0),
        key="tbl_ref",
    )

    if not sel_cats:
        st.info("계열을 선택하세요")
        return

    # Reference yields (current + 1M ago)
    ref_cur, ref_1m = {}, {}
    for tn in tenors:
        s = bond_df[(bond_df["category"] == ref_cat) & (bond_df["tenor"] == tn)].sort_values("date")
        ref_cur[tn] = s.iloc[-1]["yield"] if len(s) else np.nan
        ref_1m[tn]  = s.iloc[-22]["yield"] if len(s) >= 22 else np.nan

    rows = []
    for cat in sel_cats:
        sub = bond_df[bond_df["category"] == cat]
        row: dict = {
            "섹터": sub["sector"].iloc[0] if len(sub) else "",
            "등급": sub["rating"].iloc[0] if len(sub) else "",
        }
        for tn in tenors:
            s = bond_df[(bond_df["category"] == cat) & (bond_df["tenor"] == tn)].sort_values("date")
            cur  = s.iloc[-1]["yield"]  if len(s)    else np.nan
            cur1m= s.iloc[-22]["yield"] if len(s) >= 22 else np.nan
            row[f"y_{tn}"]  = cur
            sp  = (cur  - ref_cur[tn]) * 100 if not any(np.isnan([cur,  ref_cur[tn]])) else np.nan
            sp1m= (cur1m- ref_1m[tn])  * 100 if not any(np.isnan([cur1m,ref_1m[tn]])) else np.nan
            row[f"sp_{tn}"] = sp
            row[f"ch_{tn}"] = sp - sp1m if not any(np.isnan([sp, sp1m])) else np.nan
        rows.append(row)

    # Build HTML table
    def _fmt(v, fmt, empty="-"):
        return fmt.format(v) if not (v is None or (isinstance(v, float) and np.isnan(v))) else empty

    hdr = (
        "<style>.ct{border-collapse:collapse;width:100%;font-size:11.5px;"
        "font-family:'Apple SD Gothic Neo',sans-serif}"
        ".ct th{background:#2D3F38;color:#fff;padding:5px 8px;text-align:center;border:1px solid #1B5E20}"
        ".ct th.sub{background:#4A5E35;font-size:10px}"
        ".ct td{padding:4px 7px;text-align:center;border:1px solid #E8EDE4}"
        ".ct tr:nth-child(even) td{background:#F7F8F5}"
        ".ct .neg{color:#C62828;font-weight:600}.ct .pos{color:#2D6A4F;font-weight:600}"
        ".ct .lbl{text-align:left;font-weight:600;color:#2D3F38;background:#EEF2EA!important}"
        "</style>"
        '<table class="ct"><thead><tr>'
        '<th rowspan="2">섹터</th><th rowspan="2">등급</th>'
        '<th colspan="5">금리(%)</th>'
        '<th colspan="5">스프레드(bp)</th>'
        '<th colspan="5">전월대비 변화(bp)</th>'
        "</tr><tr>"
    )
    hdr += "".join(f'<th class="sub">{t}</th>' for _ in range(3) for t in tenors)
    hdr += "</tr></thead><tbody>"

    body = ""
    for r in rows:
        body += f'<tr><td class="lbl">{r["섹터"]}</td><td class="lbl">{r["등급"]}</td>'
        for tn in tenors:
            body += f'<td>{_fmt(r[f"y_{tn}"], "{:.2f}")}</td>'
        for tn in tenors:
            body += f'<td>{_fmt(r[f"sp_{tn}"], "{:.1f}")}</td>'
        for tn in tenors:
            v = r[f"ch_{tn}"]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                body += "<td>-</td>"
            elif v < 0:
                body += f'<td class="neg">({abs(v):.1f})</td>'
            else:
                body += f'<td class="pos">{v:.1f}</td>'
        body += "</tr>"

    st.markdown(hdr + body + "</tbody></table>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 2: Spread Chart
# ---------------------------------------------------------------------------
def _tab_spread(df: pd.DataFrame) -> None:
    bond_df  = get_bond_data(df)
    pr_df    = get_policy_rate(df)
    all_cats = _bond_cats(df)

    c1, c2, c3 = st.columns([2, 2, 1])
    cat_a = c1.selectbox(
        "계열 A", all_cats,
        index=next((i for i, c in enumerate(all_cats) if "카드채 AA" in c), 0),
        key="sp_a",
    )
    cat_b = c2.selectbox(
        "계열 B (기준)", all_cats,
        index=next((i for i, c in enumerate(all_cats) if "국고채" in c or "공사/공단채 AAA" in c), 0),
        key="sp_b",
    )
    tenor = c3.selectbox("만기", TENOR_LABELS, index=TENOR_LABELS.index("2Y"), key="sp_tenor")

    o1, o2, o3 = st.columns(3)
    show_fill   = o1.toggle("영역 채우기", value=True, key="sp_fill")
    show_avg    = o2.toggle("평균선", value=True, key="sp_avg")
    show_policy = o3.toggle("기준금리", value=not pr_df.empty, key="sp_policy",
                            disabled=pr_df.empty)

    d_start, d_end = date_range_picker(df, "sp")
    dff = bond_df[bond_df["date"].between(d_start, d_end)]

    s_a = dff[(dff["category"] == cat_a) & (dff["tenor"] == tenor)].sort_values("date")
    s_b = dff[(dff["category"] == cat_b) & (dff["tenor"] == tenor)].sort_values("date")
    merged = (
        s_a[["date", "yield"]].rename(columns={"yield": "ya"})
        .merge(s_b[["date", "yield"]].rename(columns={"yield": "yb"}), on="date")
    )
    merged["sp"] = (merged["ya"] - merged["yb"]) * 100

    if merged.empty:
        st.warning("공통 날짜 데이터 없음")
        return

    use_sec = show_policy and not pr_df.empty
    fig = make_subplots(specs=[[{"secondary_y": True}]]) if use_sec else make_subplots(specs=[[{"secondary_y": True}]])

    fill_opt = dict(fill="tozeroy", fillcolor="rgba(78,155,90,0.12)") if show_fill else {}
    fig.add_trace(
        go.Scatter(
            x=merged["date"], y=merged["sp"],
            name="스프레드(bp)",
            line=dict(color="#4E9B5A", width=1.5),
            hovertemplate="스프레드: %{y:.1f}bp<extra></extra>",
            **fill_opt,
        ),
        secondary_y=False,
    )

    if show_avg:
        avg = merged["sp"].mean()
        fig.add_trace(
            go.Scatter(
                x=[merged["date"].min(), merged["date"].max()], y=[avg, avg],
                name=f"평균 {avg:.1f}bp",
                line=dict(color="#BDBDBD", width=1.2, dash="dash"),
                hoverinfo="skip",
            ),
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(
            x=s_b["date"], y=s_b["yield"],
            name=f"{cat_b} {tenor}",
            line=dict(color="#BDBDBD", width=1.5, dash="dot"),
            hovertemplate="%{y:.3f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=s_a["date"], y=s_a["yield"],
            name=f"{cat_a} {tenor}",
            line=dict(color=DEEP_GREEN, width=2),
            hovertemplate="%{y:.3f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    if use_sec:
        _add_policy_traces(fig, pr_df, d_start, d_end, secondary_y=True)

    # --- Title shown in the app only (kept OUT of the downloaded PNG) ---
    chart_title = f"{cat_a} {tenor} 금리 및 스프레드"
    st.markdown(
        f'<div style="font-weight:700;color:{DEEP_GREEN};font-size:14px;margin:4px 0 2px">'
        f"{chart_title}</div>",
        unsafe_allow_html=True,
    )

    base_layout(fig, "", 470)  # empty layout title -> no title in export
    # Move the legend below the plot so series/legend never overlap the data
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#1A1A1A"),
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
        ),
        margin=dict(l=58, r=64, t=20, b=80),
    )
    fig.update_yaxes(title_text="스프레드(bp)", ticksuffix="bp", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="금리(%)", ticksuffix="%", secondary_y=True, showgrid=False)

    # Per-tab export config: PNG uses the (empty) layout title, so no title is baked in.
    spread_cfg = copy.deepcopy(PLOTLY_CONFIG)
    spread_cfg["toImageButtonOptions"]["filename"] = f"spread_{cat_a}_{tenor}".replace(" ", "_")
    spread_cfg["toImageButtonOptions"]["height"] = 470

    st.plotly_chart(fig, use_container_width=True, config=spread_cfg)

    last_sp = merged["sp"].iloc[-1]
    avg_sp  = merged["sp"].mean()
    pct_rk  = (merged["sp"] < last_sp).mean() * 100
    prev_1m = merged[merged["date"] <= merged["date"].max() - pd.Timedelta(days=21)]
    mom_sp  = last_sp - prev_1m["sp"].iloc[-1] if len(prev_1m) else np.nan

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("현재 스프레드", f"{last_sp:.1f}bp")
    m2.metric("기간 평균",     f"{avg_sp:.1f}bp")
    m3.metric("1M 변화",      f"{mom_sp:+.1f}bp" if not np.isnan(mom_sp) else "-")
    m4.metric("Percentile",   f"{pct_rk:.0f}%")
    m5.metric("최대/최소",    f"{merged['sp'].max():.0f}/{merged['sp'].min():.0f}bp")


# ---------------------------------------------------------------------------
# Tab 3: Curve & Change
# ---------------------------------------------------------------------------
def _single_curve(df: pd.DataFrame, cat: str, d1: pd.Timestamp, d2: pd.Timestamp) -> None:
    bond_df = get_bond_data(df)
    avail   = bond_df[bond_df["category"] == cat]["date"].unique()
    if len(avail) == 0:
        st.warning(f"데이터 없음: {cat}")
        return

    def nearest(target: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp]:
        nd = min(avail, key=lambda x: abs((x - target).days))
        return bond_df[(bond_df["category"] == cat) & (bond_df["date"] == nd)].copy(), nd

    cv1, ad1 = nearest(d1)
    cv2, ad2 = nearest(d2)
    common_t = [t for t in TENOR_LABELS if t in set(cv1["tenor"]) and t in set(cv2["tenor"])]
    if not common_t:
        st.warning("공통 만기 없음")
        return

    m1 = cv1.set_index("tenor")["yield"]
    m2 = cv2.set_index("tenor")["yield"]
    chg_bp = [(m1[t] - m2[t]) * 100 for t in common_t]
    bar_colors = ["#2D6A4F" if v >= 0 else "#C62828" for v in chg_bp]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=common_t, y=chg_bp, name="변동(bp)",
            marker_color=bar_colors, marker_line_width=0, opacity=0.8,
            text=[f"{v:+.1f}" if abs(v) >= max(abs(x) for x in chg_bp) * 0.3 else "" for v in chg_bp],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{x}: %{y:+.2f}bp<extra></extra>",
        ),
        secondary_y=False,
    )

    for curve, label, color, dash, sym in [
        (cv2, ad2.strftime("%Y-%m-%d"), "#9E9E9E", "dot", "circle-open"),
        (cv1, ad1.strftime("%Y-%m-%d"), DEEP_GREEN, "solid", "circle"),
    ]:
        cs = (curve[curve["tenor"].isin(TENOR_LABELS)]
              .assign(_ord=lambda d: d["tenor"].map(_TENOR_ORDER))
              .sort_values("_ord"))
        fig.add_trace(
            go.Scatter(
                x=cs["tenor"], y=cs["yield"], name=label,
                mode="lines+markers",
                line=dict(color=color, width=2, dash=dash),
                marker=dict(size=7, symbol=sym, line=dict(width=2, color=color)),
                hovertemplate="%{x}: %{y:.3f}%<extra></extra>",
            ),
            secondary_y=True,
        )

    max_abs = max(abs(v) for v in chg_bp) if chg_bp else 1
    fig.update_layout(
        template="plotly_white", height=400, bargap=0.3,
        font=dict(family="Apple SD Gothic Neo, Noto Sans KR, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        margin=dict(l=65, r=75, t=36, b=44),
        plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
        xaxis=dict(categoryorder="array", categoryarray=TENOR_LABELS, showgrid=False,
                   showline=True, linecolor="#BDBDBD"),
    )
    fig.update_yaxes(title_text="변동(bp)", ticksuffix="bp", secondary_y=False,
                     range=[-max_abs * 1.6, max_abs * 1.6],
                     zeroline=True, zerolinecolor="#AAAAAA", showgrid=True, gridcolor="#EEEEEE")
    fig.update_yaxes(title_text="금리(%)", ticksuffix="%", secondary_y=True,
                     showgrid=False)
    st.markdown(f"**{cat}** — {ad1.strftime('%Y-%m-%d')} vs {ad2.strftime('%Y-%m-%d')}")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def _tab_curve(df: pd.DataFrame) -> None:
    bond_df  = get_bond_data(df)
    all_cats = _bond_cats(df)
    avail    = sorted(bond_df["date"].dropna().unique(), reverse=True)
    min_d    = pd.Timestamp(avail[-1]).date()
    max_d    = pd.Timestamp(avail[0]).date()

    st.caption("영업일이 아닌 날은 가장 가까운 영업일로 자동 조정됩니다.")
    c1, c2, c3 = st.columns([2, 2, 1])
    d1 = pd.Timestamp(c1.date_input("기준일 (최신)", value=max_d, min_value=min_d, max_value=max_d, key="cv_d1"))
    d2 = pd.Timestamp(c2.date_input("비교일 (이전)", value=max(max_d - datetime.timedelta(days=30), min_d),
                                     min_value=min_d, max_value=max_d, key="cv_d2"))
    n_panels = c3.radio("패널", [1, 2], index=1, horizontal=True, key="cv_n")

    if d1 < d2:
        d1, d2 = d2, d1
    if d1 == d2:
        st.warning("두 날짜가 같습니다.")
        return

    defaults = [
        next((c for c in all_cats if "공사/공단채 AAA" in c), all_cats[0]),
        next((c for c in all_cats if "카드채 AA" in c), all_cats[min(1, len(all_cats) - 1)]),
    ]
    cols = st.columns(n_panels)
    for i in range(n_panels):
        with cols[i]:
            cat = st.selectbox(
                f"계열 {i+1}", all_cats,
                index=all_cats.index(defaults[i]) if defaults[i] in all_cats else 0,
                key=f"cv_cat_{i}",
            )
            _single_curve(df, cat, d1, d2)


# ---------------------------------------------------------------------------
# Tab 4: Time Series
# ---------------------------------------------------------------------------
def _tab_timeseries(df: pd.DataFrame) -> None:
    bond_df  = get_bond_data(df)
    pr_df    = get_policy_rate(df)
    all_cats = _bond_cats(df)

    c1, c2 = st.columns([3, 1])
    ts_cats  = c1.multiselect("계열 (최대 6개)", all_cats, default=all_cats[:3], max_selections=6, key="ts_cats")
    ts_tenor = c2.selectbox("만기", TENOR_LABELS, index=TENOR_LABELS.index("3Y"), key="ts_tenor")

    show_pr = st.toggle("기준금리 오버레이", value=not pr_df.empty, key="ts_pr", disabled=pr_df.empty)
    d_start, d_end = date_range_picker(df, "ts")

    dff      = bond_df[bond_df["date"].between(d_start, d_end)]
    use_sec  = show_pr and not pr_df.empty
    fig      = make_subplots(specs=[[{"secondary_y": True}]])

    for i, cat in enumerate(ts_cats):
        s = dff[(dff["category"] == cat) & (dff["tenor"] == ts_tenor)]
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s["date"], y=s["yield"],
                name=f"{cat} ({ts_tenor})",
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                hovertemplate=f"{cat}: %{{y:.3f}}%<extra></extra>",
            ),
            secondary_y=False,
        )

    if use_sec:
        _add_policy_traces(fig, pr_df, d_start, d_end, secondary_y=True)

    base_layout(fig, f"금리 시계열 | {ts_tenor}", 430)
    fig.update_yaxes(title_text="금리(%)", ticksuffix="%", secondary_y=False)
    if use_sec:
        fig.update_yaxes(title_text="기준금리(%)", ticksuffix="%", secondary_y=True,
                         showgrid=False, tickfont=dict(color=DEEP_GREEN))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # Bar: latest levels
    lv = [
        {"계열": cat, "금리": dff[(dff["category"] == cat) & (dff["tenor"] == ts_tenor)].iloc[-1]["yield"]}
        for cat in ts_cats
        if not dff[(dff["category"] == cat) & (dff["tenor"] == ts_tenor)].empty
    ]
    if lv:
        lv_df = pd.DataFrame(lv).sort_values("금리")
        fig2 = go.Figure(go.Bar(
            x=lv_df["금리"], y=lv_df["계열"], orientation="h",
            marker_color=DEEP_GREEN,
            text=[f"{v:.3f}%" for v in lv_df["금리"]], textposition="outside",
        ))
        base_layout(fig2, f"최신 금리 비교 | {ts_tenor}", max(200, len(lv) * 44 + 80))
        fig2.update_xaxes(ticksuffix="%")
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("---")
    st.markdown("##### 만기별 금리 추이")
    c1, c2 = st.columns(2)
    mt_cat    = c1.selectbox("계열", all_cats, key="mt_cat")
    mt_tenors = c2.multiselect("만기", TENOR_LABELS, default=["1Y", "2Y", "3Y", "5Y"], key="mt_tenors")
    d_s2, d_e2 = date_range_picker(df, "mt")
    dff2 = bond_df[bond_df["date"].between(d_s2, d_e2)]
    fig3 = go.Figure()
    for i, tn in enumerate(mt_tenors or ["1Y", "3Y"]):
        s = dff2[(dff2["category"] == mt_cat) & (dff2["tenor"] == tn)]
        if not s.empty:
            fig3.add_trace(go.Scatter(x=s["date"], y=s["yield"], name=tn,
                line=dict(color=_GREEN_SHADES[i % len(_GREEN_SHADES)], width=1.8),
                hovertemplate=f"{tn}: %{{y:.3f}}%<extra></extra>"))
    base_layout(fig3, f"만기별 금리 | {mt_cat}", 400)
    fig3.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG)


# ---------------------------------------------------------------------------
# Tab 5: Policy Rate
# ---------------------------------------------------------------------------
def _tab_policy(df: pd.DataFrame) -> None:
    pr_df    = get_policy_rate(df)
    bond_df  = get_bond_data(df)
    all_cats = _bond_cats(df)

    if pr_df.empty:
        st.info("기준금리 데이터 없음 (헤더에 {국가}:기준금리 형식 필요)")
        return

    _policy_banner(df)
    st.markdown("---")

    d_start, d_end = date_range_picker(df, "pr")
    dff_pr = pr_df[pr_df["date"].between(d_start, d_end)]

    fig = go.Figure()
    for cat in sorted(pr_df["category"].unique()):
        s = dff_pr[dff_pr["category"] == cat].sort_values("date")
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["date"], y=s["yield"], name=f"{s.iloc[0]['rating']} 기준금리",
            line=dict(color=DEEP_GREEN, width=2.5, dash="dashdot"),
            line_shape="hv", fill="tozeroy", fillcolor="rgba(45,63,56,0.07)",
            hovertemplate="%{y:.2f}%<extra></extra>",
        ))
    base_layout(fig, "기준금리 추이", 360)
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("---")
    st.markdown("#### 기준금리 vs 채권 금리")
    c1, c2 = st.columns([3, 1])
    cmp_cats  = c1.multiselect("비교 채권 계열 (최대 4개)", all_cats,
        default=[c for c in all_cats if "국고채" in c or "공사/공단채 AAA" in c][:2],
        max_selections=4, key="pr_cats")
    cmp_tenor = c2.selectbox("만기", TENOR_LABELS, index=TENOR_LABELS.index("3Y"), key="pr_tenor")

    dff_b = bond_df[bond_df["date"].between(d_start, d_end)]
    fig2  = go.Figure()
    for cat in sorted(pr_df["category"].unique()):
        s = dff_pr[dff_pr["category"] == cat].sort_values("date")
        if s.empty:
            continue
        fig2.add_trace(go.Scatter(
            x=s["date"], y=s["yield"], name=f"{s.iloc[0]['rating']} 기준금리",
            line=dict(color=DEEP_GREEN, width=2.5, dash="dashdot"), line_shape="hv",
            hovertemplate="%{y:.2f}%<extra></extra>",
        ))
    for i, cat in enumerate(cmp_cats):
        s = dff_b[(dff_b["category"] == cat) & (dff_b["tenor"] == cmp_tenor)].sort_values("date")
        if s.empty:
            continue
        fig2.add_trace(go.Scatter(
            x=s["date"], y=s["yield"], name=f"{cat} {cmp_tenor}",
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            hovertemplate="%{y:.3f}%<extra></extra>",
        ))
    base_layout(fig2, f"기준금리 vs 채권 금리 ({cmp_tenor})", 420)
    fig2.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render(df: pd.DataFrame) -> None:
    st.header("Market View")

    bond_df = get_bond_data(df)
    pr_df   = get_policy_rate(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("데이터 시작", df["date"].min().strftime("%Y-%m-%d"))
    c2.metric("데이터 종료", df["date"].max().strftime("%Y-%m-%d"))
    c3.metric("채권 계열", f"{bond_df['category'].nunique()}개")
    if not pr_df.empty:
        row = pr_df.sort_values("date").iloc[-1]
        c4.metric(f"{row['rating']} 기준금리", f"{row['yield']:.2f}%")
    else:
        c4.metric("기준금리", "데이터 없음")

    st.markdown("---")

    if not pr_df.empty:
        tabs = st.tabs(["금리·스프레드 변동표", "크레딧 스프레드", "커브·변동 비교", "금리 시계열", "기준금리"])
        with tabs[0]: _tab_summary(df)
        with tabs[1]: _tab_spread(df)
        with tabs[2]: _tab_curve(df)
        with tabs[3]: _tab_timeseries(df)
        with tabs[4]: _tab_policy(df)
    else:
        tabs = st.tabs(["금리·스프레드 변동표", "크레딧 스프레드", "커브·변동 비교", "금리 시계열"])
        with tabs[0]: _tab_summary(df)
        with tabs[1]: _tab_spread(df)
        with tabs[2]: _tab_curve(df)
        with tabs[3]: _tab_timeseries(df)