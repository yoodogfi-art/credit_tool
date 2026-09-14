"""Credit Research Engine — main entry point."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from assets.styles import CSS, DEEP_GREEN
from data.loader import load_excel

st.set_page_config(
    page_title="Credit Research Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div style="padding:16px 0 8px;text-align:center">'
        f'<div style="font-weight:800;font-size:15px;color:{DEEP_GREEN}">Credit Research</div>'
        f'<div style="font-size:11px;color:#9E9E9E">채권 크레딧 분석 엔진</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("##### 데이터 업로드")
    uploaded = st.file_uploader(
        "Excel 파일",
        type=["xlsx", "xls"],
        help="시가평가 3사평균 Wide-format Excel",
        label_visibility="collapsed",
    )
    if uploaded:
        st.success(f"{uploaded.name}")
    else:
        st.info("Excel 파일을 업로드하세요")

    st.divider()
    st.markdown("##### 메뉴")
    page = st.radio(
        "페이지",
        ["Market View", "Sector Matrix", "Signal Dashboard"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("v0.3")

# ---------------------------------------------------------------------------
# Landing page (no file)
# ---------------------------------------------------------------------------
if uploaded is None:
    st.markdown(
        f'<div style="text-align:center;padding:60px 20px">'
        f'<h1 style="color:{DEEP_GREEN};font-size:2rem">크레딧 리서치 엔진</h1>'
        f'<p style="color:#616161;font-size:15px">'
        f'데이터 업로드 &rarr; 자동 분석 &rarr; Score &rarr; OW/NW/UW</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    for col, title, desc in [
        (c1, "Market View",      "금리 시계열, 스프레드, 커브, 월간 변화 시각화"),
        (c2, "Sector Matrix",    "섹터 x 등급 x 만기 히트맵 + 자동 OW/NW/UW"),
        (c3, "Signal Dashboard", "스프레드 / Z-score 매트릭스 + Duration Spread"),
    ]:
        col.markdown(
            f'<div style="border:1px solid #DDE4D8;border-radius:6px;padding:18px;background:#F7F8F5">'
            f'<div style="font-weight:700;color:{DEEP_GREEN};margin-bottom:6px">{title}</div>'
            f'<div style="font-size:13px;color:#555">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.stop()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with st.spinner("데이터 파싱 중..."):
    try:
        df, load_warnings = load_excel(uploaded.getvalue())
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")
        st.stop()

for msg in load_warnings:
    if msg.startswith("__toast__"):
        st.toast(msg[len("__toast__"):])
    else:
        st.warning(msg)

with st.sidebar:
    st.markdown(
        f'<div style="background:#EEF4EB;border-radius:5px;padding:8px 10px;font-size:11px;color:#333">'
        f'{df["date"].min().strftime("%Y-%m-%d")} ~ {df["date"].max().strftime("%Y-%m-%d")}<br>'
        f'계열 {df["category"].nunique()}개 &nbsp;|&nbsp; {len(df):,}행'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("##### 조회 기간")
    import datetime as _dt
    _min_d = df["date"].min().date()
    _max_d = df["date"].max().date()
    _default_start = max(_max_d - _dt.timedelta(days=365), _min_d)
    g_start = st.date_input("시작일", value=_default_start, min_value=_min_d, max_value=_max_d, key="sb_date_start")
    g_end   = st.date_input("종료일", value=_max_d,         min_value=_min_d, max_value=_max_d, key="sb_date_end")
    if g_start > g_end:
        st.warning("시작일이 종료일보다 늦습니다.")

# ---------------------------------------------------------------------------
# Route to page
# ---------------------------------------------------------------------------
if page == "Market View":
    from pages.market_view import render
elif page == "Sector Matrix":
    from pages.sector_matrix import render
elif page == "Signal Dashboard":
    from pages.signal_dashboard import render

render(df)