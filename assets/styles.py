DEEP_GREEN  = "#2D3F38"
LEAF_GREEN  = "#4E9B5A"
OLIVE       = "#4A5E35"
CORAL       = "#C0392B"
GRAY        = "#8A9E96"

# Goldman Sachs-style grayish gold for policy rate / accent
GS_GOLD           = "#B5A46A"
GS_GOLD_FILL      = "rgba(181,164,106,0.10)"

CHART_COLORS = [DEEP_GREEN, GS_GOLD, OLIVE, "#6B7B9A", "#9A7085", CORAL]

# Heatmap: low value (low rate / tight spread) = green, high = red
# For rate/spread level: lower is better (bull) = green
HEATMAP_RATE   = [[0, "#2D6A4F"], [0.5, "#F7F8F5"], [1, "#8B1A1A"]]   # low=green, high=red
HEATMAP_DIVERG = [[0, "#2D6A4F"], [0.5, "#F7F8F5"], [1, "#8B1A1A"]]   # spread: tight=green, wide=red
# Legacy alias kept for any direct reference
HEATMAP_GREEN  = HEATMAP_RATE

PLOTLY_TEMPLATE = "plotly_white"

POLICY_COLOR      = GS_GOLD
POLICY_FILL_COLOR = GS_GOLD_FILL

CSS = """
<style>
[data-testid="stSidebarNav"] { display: none !important; }
html, body, [class*="css"] {
    font-family: 'Apple SD Gothic Neo', 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    color: #212121;
}
[data-testid="stSidebar"] {
    background-color: #F7F8F5;
    border-right: 1px solid #DDE4D8;
}
h1 { color: #2D3F38 !important; font-weight: 700 !important; }
h2 { color: #2D3F38 !important; font-weight: 600 !important; }
h3 { color: #4A5E35 !important; font-weight: 600 !important; }
.stButton > button {
    background-color: #4E9B5A !important;
    color: white !important;
    border-radius: 4px !important;
    border: none !important;
    font-size: 13px !important;
}
.stButton > button:hover { background-color: #2D3F38 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid #DDE4D8; }
.stTabs [data-baseweb="tab"] {
    border-radius: 4px 4px 0 0;
    padding: 6px 16px;
    font-size: 13px;
    color: #6B7B6E;
}
.stTabs [aria-selected="true"] {
    background-color: #F7F8F5 !important;
    color: #2D3F38 !important;
    border-bottom: 2px solid #4E9B5A !important;
    font-weight: 600 !important;
}
hr { border-color: #DDE4D8 !important; margin: 12px 0 !important; }
[data-testid="metric-container"] {
    background: #F7F8F5;
    border: 1px solid #DDE4D8;
    border-radius: 4px;
    padding: 10px 14px;
}
/* Hide the previous page's stale/ghost content during a rerun instead of
   fading it in place — this app swaps whole unrelated pages, so a dimmed
   leftover of the old page reads as broken rather than "loading". */
[data-testid="stElementContainer"][data-stale="true"],
[data-testid="stVerticalBlock"][data-stale="true"] {
    display: none !important;
}
</style>
"""