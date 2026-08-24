# -*- coding: utf-8 -*-
"""
NBA Game & Player Assistant Agent  (v3 — UI Redesign)
------------------------------------------------------
Gelistirmeler:
  v2: Dinamik sezon, API retry, AI Router fix
  v3: Tam UI yenileme — custom NBA dark theme, gradient
      basliklar, styled kartlar, renkli tablolar, hero banner

Veri    : nba_api (PlayerGameLog, ShotChartDetail)
Gorsel  : Streamlit + Matplotlib
Yerel AI: Ollama Chat API — qwen2.5:7b
"""

import json
import re
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np          # noqa: F401
import pandas as pd
import requests
import streamlit as st
from nba_api.stats.endpoints import playergamelog, shotchartdetail
from nba_api.stats.static import players

# ══════════════════════════════════════════════════════════
# SAYFA KONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NBA Local AI Assistant",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# CUSTOM CSS — NBA DARK THEME
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Genel arka plan ── */
.stApp { background-color: #0a0e1a; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a 0%, #1a2744 100%); border-right: 1px solid #1e3a5f; }

/* ── Tum yazi ── */
html, body, [class*="css"] { color: #e8eaf0; font-family: 'Segoe UI', sans-serif; }

/* ── Basliklar ── */
h1 { background: linear-gradient(135deg, #c8a84b 0%, #f5d060 50%, #c8a84b 100%);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     font-size: 2.2rem !important; font-weight: 800 !important; letter-spacing: -0.5px; }
h2 { color: #c8a84b !important; font-weight: 700 !important; font-size: 1.3rem !important; }
h3 { color: #90b4d4 !important; font-weight: 600 !important; }

/* ── Metric kartlari ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827 0%, #1e2d45 100%);
    border: 1px solid #2a4a6b;
    border-radius: 14px;
    padding: 18px 20px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4), inset 0 1px 0 rgba(200,168,75,0.15);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(200,168,75,0.2), inset 0 1px 0 rgba(200,168,75,0.2);
}
[data-testid="stMetricLabel"]  { color: #8ba8c4 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"]  { color: #f5d060 !important; font-size: 2rem !important; font-weight: 800 !important; }

/* ── Butonlar ── */
.stButton > button {
    background: linear-gradient(135deg, #c8a84b 0%, #f5d060 100%);
    color: #0a0e1a !important; font-weight: 700; border: none;
    border-radius: 10px; padding: 10px 24px;
    box-shadow: 0 4px 15px rgba(200,168,75,0.35);
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(200,168,75,0.5);
    background: linear-gradient(135deg, #d4b55a 0%, #fce070 100%);
}
.stButton > button[kind="secondary"] {
    background: #1e2d45 !important; color: #c8a84b !important;
    border: 1px solid #2a4a6b !important;
}

/* ── Form submit butonlari ── */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #c8a84b 0%, #f5d060 100%) !important;
    color: #0a0e1a !important; font-weight: 700 !important;
    border-radius: 10px !important; border: none !important;
}

/* ── Input alanlari ── */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: #111827 !important; color: #e8eaf0 !important;
    border: 1px solid #2a4a6b !important; border-radius: 10px !important;
    padding: 10px 14px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #c8a84b !important;
    box-shadow: 0 0 0 2px rgba(200,168,75,0.2) !important;
}

/* ── Selectbox & Radio ── */
.stSelectbox > div > div,
.stRadio > div { background: #111827 !important; border-radius: 10px !important; }
.stSelectbox > div > div { border: 1px solid #2a4a6b !important; }

/* ── Slider ── */
.stSlider > div > div > div > div { background: #c8a84b !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #1e3a5f;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8ba8c4 !important;
    border-radius: 9px !important;
    font-weight: 600;
    padding: 10px 20px !important;
    border: none !important;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #c8a84b 0%, #f5d060 100%) !important;
    color: #0a0e1a !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid #1e3a5f; }
[data-testid="stDataFrame"] > div { background: #111827 !important; }

/* ── Info / Warning / Error kutusu ── */
.stAlert { border-radius: 12px !important; border-left-width: 4px !important; }
[data-testid="stInfoIcon"]    ~ div { background: rgba(30,58,95,0.5) !important; }

/* ── Sidebar elementleri ── */
[data-testid="stSidebar"] .stRadio label { color: #c8d8e8 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #8ba8c4 !important; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.8px; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #c8a84b !important; }

/* ── Divider ── */
hr { border-color: #1e3a5f !important; margin: 1.5rem 0 !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #111827 !important; border: 1px solid #1e3a5f !important;
    border-radius: 10px !important; color: #90b4d4 !important; font-weight: 600;
}
.streamlit-expanderContent { background: #0d1520 !important; border: 1px solid #1e3a5f !important; border-top: none !important; border-radius: 0 0 10px 10px !important; }

/* ── Caption / small text ── */
.stCaption, small { color: #5a7a9a !important; }

/* ── Bar chart ── */
[data-testid="stVegaLiteChart"] { border-radius: 12px; overflow: hidden; }

/* ── Kart kutusu (custom HTML) ── */
.nba-card {
    background: linear-gradient(135deg, #111827 0%, #1a2744 100%);
    border: 1px solid #2a4a6b;
    border-radius: 16px;
    padding: 20px 24px;
    margin: 8px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.nba-hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a3050 40%, #0d1b2a 100%);
    border: 1px solid #2a4a6b;
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(200,168,75,0.2);
    text-align: center;
}
.zone-good   { color: #3ddc84; font-weight: 700; }
.zone-mid    { color: #f5d060; font-weight: 700; }
.zone-bad    { color: #ff6b6b; font-weight: 700; }
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-gold   { background: rgba(200,168,75,0.2); color: #f5d060; border: 1px solid #c8a84b; }
.badge-blue   { background: rgba(30,58,95,0.5);   color: #90b4d4; border: 1px solid #2a4a6b; }
.badge-green  { background: rgba(61,220,132,0.15); color: #3ddc84; border: 1px solid #3ddc84; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# GENEL AYARLAR
# ══════════════════════════════════════════════════════════
OLLAMA_CHAT_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL      = "qwen2.5:7b"
REQUEST_TIMEOUT   = 120
AVAILABLE_SEASONS = ["2024-25", "2023-24", "2022-23", "2021-22"]

# ══════════════════════════════════════════════════════════
# YARDIMCI HTML FONKSIYONLARI
# ══════════════════════════════════════════════════════════

def hero_banner(player_name: str, season: str, n_games: int):
    st.markdown(f"""
    <div class="nba-hero">
        <div style="font-size:3rem; margin-bottom:6px;">🏀</div>
        <h1 style="margin:0; font-size:2rem;">{player_name}</h1>
        <div style="margin-top:10px; display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">
            <span class="badge badge-gold">📅 {season}</span>
            <span class="badge badge-blue">🎮 Son {n_games} Maç</span>
            <span class="badge badge-green">⚡ Yerel AI Aktif</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def section_header(icon: str, title: str):
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; margin:20px 0 12px 0;">
        <span style="font-size:1.5rem;">{icon}</span>
        <h2 style="margin:0;">{title}</h2>
    </div>
    """, unsafe_allow_html=True)


def styled_ai_output(text: str):
    """AI analizini bolumlere gore renkli badge'lerle goster."""
    st.markdown(f"""
    <div class="nba-card" style="border-left:4px solid #c8a84b;">
        <div style="white-space:pre-wrap; line-height:1.7; color:#d0dae8;">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def zone_table_html(df: pd.DataFrame) -> str:
    """Bolgesel istatistik tablosunu renkli HTML olarak uretir."""
    if df.empty:
        return "<p style='color:#5a7a9a;'>Veri yok</p>"

    rows = ""
    for _, r in df.iterrows():
        pct = r["Yuzde"]
        if pct >= 55:
            cls, bar_color = "zone-good", "#3ddc84"
        elif pct >= 43:
            cls, bar_color = "zone-mid",  "#f5d060"
        else:
            cls, bar_color = "zone-bad",  "#ff6b6b"

        bar_w = min(int(pct * 1.4), 100)
        rows += f"""
        <tr>
            <td style="padding:10px 12px; color:#c8d8e8; font-weight:600;">{r['Bolge']}</td>
            <td style="padding:10px 12px; color:#8ba8c4; text-align:center;">{r['Denenen']}</td>
            <td style="padding:10px 12px; color:#8ba8c4; text-align:center;">{r['Isabetli']}</td>
            <td style="padding:10px 12px; text-align:center;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <div style="flex:1; background:#1e2d45; border-radius:4px; height:8px; min-width:60px;">
                        <div style="width:{bar_w}%; background:{bar_color}; height:8px; border-radius:4px;"></div>
                    </div>
                    <span class="{cls}">%{pct}</span>
                </div>
            </td>
        </tr>"""

    return f"""
    <table style="width:100%; border-collapse:collapse; background:#111827; border-radius:12px; overflow:hidden;">
        <thead>
            <tr style="background:#1e2d45; border-bottom:1px solid #2a4a6b;">
                <th style="padding:10px 12px; color:#8ba8c4; text-align:left; font-size:0.78rem; text-transform:uppercase; letter-spacing:1px;">Bölge</th>
                <th style="padding:10px 12px; color:#8ba8c4; text-align:center; font-size:0.78rem; text-transform:uppercase;">Deneme</th>
                <th style="padding:10px 12px; color:#8ba8c4; text-align:center; font-size:0.78rem; text-transform:uppercase;">İsabet</th>
                <th style="padding:10px 12px; color:#8ba8c4; text-align:center; font-size:0.78rem; text-transform:uppercase;">İsabet %</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>"""


# ══════════════════════════════════════════════════════════
# VERI CEKME — CACHE + RETRY
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def find_player_id(player_name: str):
    if not player_name or not player_name.strip():
        return None, None
    try:
        matches = players.find_players_by_full_name(player_name.strip())
    except Exception:
        return None, None
    if not matches:
        return None, None
    active = [p for p in matches if p.get("is_active")]
    target = active[0] if active else matches[0]
    return target["id"], target["full_name"]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_game_log(player_id: int, season: str):
    last_error = None
    for attempt in range(1, 3):
        try:
            log = playergamelog.PlayerGameLog(
                player_id=player_id, season=season, timeout=90
            )
            df = log.get_data_frames()[0]
            if df.empty:
                return df, None
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            df = df.sort_values("GAME_DATE").reset_index(drop=True)
            return df, None
        except requests.exceptions.Timeout:
            last_error = f"NBA API zaman asimi (deneme {attempt}/2)."
        except requests.exceptions.ConnectionError:
            last_error = "NBA API baglantisi kurulamadi."
        except Exception as exc:
            last_error = f"Mac logu hatasi: {exc}"
        if attempt < 2:
            time.sleep(1.5)
    return pd.DataFrame(), last_error


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_shot_chart(player_id: int, season: str):
    last_error = None
    for attempt in range(1, 3):
        try:
            sc = shotchartdetail.ShotChartDetail(
                team_id=0, player_id=player_id, season_nullable=season,
                season_type_all_star="Regular Season",
                context_measure_simple="FGA", timeout=90,
            )
            return sc.get_data_frames()[0], None
        except requests.exceptions.Timeout:
            last_error = f"Sut haritasi zaman asimi (deneme {attempt}/2)."
        except requests.exceptions.ConnectionError:
            last_error = "NBA API baglantisi kurulamadi."
        except Exception as exc:
            last_error = f"Sut haritasi hatasi: {exc}"
        if attempt < 2:
            time.sleep(1.5)
    return pd.DataFrame(), last_error


# ══════════════════════════════════════════════════════════
# SAHA CIZIMI
# ══════════════════════════════════════════════════════════

def draw_court(ax=None, color="white", lw=1.6):
    if ax is None:
        ax = plt.gca()
    for el in [
        patches.Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False),
        patches.Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color),
        patches.Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False),
        patches.Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False),
        patches.Arc((0, 142.5), 120, 120, theta1=0,   theta2=180, linewidth=lw, color=color),
        patches.Arc((0, 142.5), 120, 120, theta1=180, theta2=360, linewidth=lw,
                    color=color, linestyle="dashed"),
        patches.Arc((0, 0),      80,  80, theta1=0,   theta2=180, linewidth=lw, color=color),
        patches.Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color),
        patches.Rectangle(( 220, -47.5), 0, 140, linewidth=lw, color=color),
        patches.Arc((0, 0),     475, 475, theta1=22,  theta2=158, linewidth=lw, color=color),
        patches.Arc((0, 422.5), 120, 120, theta1=180, theta2=0,   linewidth=lw, color=color),
        patches.Rectangle((-250, -47.5), 500, 470, linewidth=lw, color=color, fill=False),
    ]:
        ax.add_patch(el)
    ax.set_xlim(-260, 260); ax.set_ylim(-60, 430)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#0a0e1a")
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def plot_shot_chart(shot_df, title="", mode="Noktasal"):
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    fig.patch.set_facecolor("#0d1520")
    ax.set_facecolor("#0a0e1a")
    # Hafif degrade zemin efekti
    ax.imshow(
        [[0.06, 0.08], [0.04, 0.06]],
        extent=[-260, 260, -60, 430], aspect="auto", alpha=0.3,
        cmap="Blues", zorder=0,
    )
    draw_court(ax, color="#2a4a6b", lw=1.8)

    if shot_df.empty:
        ax.text(0, 200, "Şut verisi bulunamadı", color="#5a7a9a",
                ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc="#111827", ec="#2a4a6b", alpha=0.9))
    elif mode == "Is Haritasi":
        hb = ax.hexbin(shot_df["LOC_X"], shot_df["LOC_Y"],
                       gridsize=28, extent=(-260, 260, -60, 430),
                       cmap="inferno", mincnt=1, alpha=0.88, zorder=2)
        cbar = fig.colorbar(hb, ax=ax, shrink=0.65, pad=0.02)
        cbar.set_label("Şut Yoğunluğu", color="#8ba8c4", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="#8ba8c4")
        plt.setp(plt.getp(cbar.ax, "yticklabels"), color="#8ba8c4", fontsize=8)
        cbar.outline.set_edgecolor("#1e3a5f")
    else:
        made   = shot_df[shot_df["SHOT_MADE_FLAG"] == 1]
        missed = shot_df[shot_df["SHOT_MADE_FLAG"] == 0]
        ax.scatter(missed["LOC_X"], missed["LOC_Y"],
                   c="#ff6b6b", marker="x", s=28, alpha=0.65,
                   linewidths=1.2, label=f"Kaçan ({len(missed)})", zorder=3)
        ax.scatter(made["LOC_X"], made["LOC_Y"],
                   c="#3ddc84", marker="o", s=26, alpha=0.82,
                   edgecolors="white", linewidths=0.4,
                   label=f"İsabetli ({len(made)})", zorder=4)
        leg = ax.legend(loc="upper right", facecolor="#111827", labelcolor="#c8d8e8",
                        fontsize=8.5, framealpha=0.95, edgecolor="#2a4a6b")
        leg.get_frame().set_linewidth(1)

    ax.set_title(title or "Şut Dağılımı", color="#c8a84b", fontsize=13,
                 fontweight="bold", pad=12)
    fig.tight_layout(pad=0.5)
    return fig


def compute_zone_stats(shot_df) -> pd.DataFrame:
    if shot_df.empty:
        return pd.DataFrame(columns=["Bolge", "Denenen", "Isabetli", "Yuzde"])
    zone_map = {
        "Restricted Area":       "Boyali Alan",
        "In The Paint (Non-RA)": "Boyali Alan",
        "Mid-Range":             "Orta Mesafe",
        "Left Corner 3":         "3 Sayi",
        "Right Corner 3":        "3 Sayi",
        "Above the Break 3":     "3 Sayi",
        "Backcourt":             "Diger",
    }
    df = shot_df.copy()
    df["ZONE_TR"] = df["SHOT_ZONE_BASIC"].map(zone_map).fillna("Diger")
    g = (
        df.groupby("ZONE_TR")
          .agg(Denenen=("SHOT_MADE_FLAG", "count"),
               Isabetli=("SHOT_MADE_FLAG", "sum"))
          .reset_index()
    )
    g["Yuzde"] = (g["Isabetli"] / g["Denenen"] * 100).round(1)
    g = g.rename(columns={"ZONE_TR": "Bolge"})
    order = {"Boyali Alan": 0, "Orta Mesafe": 1, "3 Sayi": 2, "Diger": 3}
    g["_ord"] = g["Bolge"].map(order).fillna(9)
    return g.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


# ══════════════════════════════════════════════════════════
# OLLAMA AI
# ══════════════════════════════════════════════════════════

def check_ollama_alive() -> bool:
    try:
        return requests.get("http://localhost:11434/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


def _parse_json_safe(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def route_user_query(user_text: str) -> dict:
    system_prompt = (
        "You are an NLP router for an NBA statistics assistant. "
        "Extract the full NBA player name and the specific question. "
        "Return ONLY valid JSON: "
        '{"player_name": "Full Name", "clean_question": "..."}\n'
        "If no player, player_name=null. No extra text.\n"
        'Ex: "LeBron son mac?" -> {"player_name":"LeBron James","clean_question":"Son mac performansi?"}'
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_text},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 120},
    }
    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=20)
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "").strip()
        parsed = _parse_json_safe(raw)
        if parsed and "player_name" in parsed and "clean_question" in parsed:
            return parsed
    except requests.exceptions.Timeout:
        st.toast("AI Router zaman asimina ugradi.", icon="⚠️")
    except requests.exceptions.ConnectionError:
        st.toast("Ollama baglantisi kurulamadi.", icon="🔴")
    except Exception as exc:
        st.toast(f"AI Router hatasi: {exc}", icon="⚠️")
    return {"player_name": None, "clean_question": user_text}


def query_ollama_analysis(player_name, game_log, zone_stats, user_question) -> str:
    recent = game_log.tail(5)
    lines = []
    for _, row in recent.iterrows():
        d = row["GAME_DATE"].strftime("%d.%m") if hasattr(row["GAME_DATE"], "strftime") else str(row["GAME_DATE"])
        lines.append(
            f"- {d} ({row['MATCHUP']}): {row['PTS']} Sayi, "
            f"{row['REB']} Rib, {row['AST']} Ast, FG%{row['FG_PCT']*100:.0f}"
        )
    zone_lines = (
        [f"- {r['Bolge']}: {r['Isabetli']}/{r['Denenen']} (%{r['Yuzde']})"
         for _, r in zone_stats.iterrows()]
        if not zone_stats.empty else ["Veri yok"]
    )
    system_prompt = (
        "Sen uzman bir NBA taktik analistisin. Kullanicinin sorusunu verilen istatistiklere "
        "dayanarak kisa, net ve akici TURKCE ile yanitla. Tam olarak 3 baslikta yaz: "
        "1) Form Durumu  2) Sut & Skor Tehdidi  3) Taktiksel Oneri. "
        "Her baslik 2-3 cumle. Tekrara dusme."
    )
    user_prompt = (
        f"Oyuncu: {player_name}\n\nSon 5 Mac:\n" + "\n".join(lines) +
        "\n\nBolgesel Sut %:\n" + "\n".join(zone_lines) +
        f"\n\nKullanici Sorusu: {user_question}"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user",   "content": user_prompt}],
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.85, "num_predict": 350},
    }
    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except requests.exceptions.Timeout:
        return f"⏱️ AI zaman asimina ugradi ({REQUEST_TIMEOUT}s). Tekrar deneyin."
    except requests.exceptions.ConnectionError:
        return "🔌 Ollama kapali veya erisilemez (localhost:11434)."
    except Exception as exc:
        return f"❌ AI Baglanti Hatasi: {exc}"


def query_ollama_comparison(p1_name, p1, p2_name, p2) -> str:
    system_prompt = (
        "Sen profesyonel bir NBA scout ve analistisin. Iki oyuncuyu kiyaslayan "
        "kisa, net ve akici TURKCE scouting raporu yaz. 3 baslik: "
        "1) Skor & Verimlilik  2) Sut Haritasi & Alan Hakimiyeti  3) Koc Esleme Notu. "
        "Her baslik 2-3 cumle."
    )
    user_prompt = (
        f"OYUNCU 1: {p1_name}\n"
        f"Ortalamalar: {p1['pts']:.1f} Sayi, {p1['reb']:.1f} Rib, {p1['ast']:.1f} Ast, %{p1['fg']:.1f} FG\n"
        f"Sut Bolgeleri: {p1['zones']}\n\n"
        f"OYUNCU 2: {p2_name}\n"
        f"Ortalamalar: {p2['pts']:.1f} Sayi, {p2['reb']:.1f} Rib, {p2['ast']:.1f} Ast, %{p2['fg']:.1f} FG\n"
        f"Sut Bolgeleri: {p2['zones']}\n\nTaktiksel karsilastirma:"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user",   "content": user_prompt}],
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.85, "num_predict": 400},
    }
    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except requests.exceptions.Timeout:
        return f"⏱️ AI zaman asimina ugradi ({REQUEST_TIMEOUT}s). Tekrar deneyin."
    except requests.exceptions.ConnectionError:
        return "🔌 Ollama kapali veya erisilemez (localhost:11434)."
    except Exception as exc:
        return f"❌ AI Baglanti Hatasi: {exc}"


# ══════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════
_DEFAULTS = {
    "active_player":     "Alperen Sengun",
    "h2h_p1":           "Alperen Sengun",
    "h2h_p2":           "Nikola Jokic",
    "num_games":         10,
    "chart_mode":       "Noktasal",
    "season":           "2023-24",
    "last_ai_analysis":  "",
    "last_h2h_analysis": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0 8px 0;">
        <div style="font-size:2.5rem;">🏀</div>
        <div style="font-size:1.1rem; font-weight:800; color:#c8a84b; letter-spacing:0.5px;">NBA LOCAL AI</div>
        <div style="font-size:0.72rem; color:#5a7a9a; margin-top:2px; text-transform:uppercase; letter-spacing:1px;">Assistant v3</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    selected_season = st.selectbox(
        "🗓️ Sezon",
        options=AVAILABLE_SEASONS,
        index=AVAILABLE_SEASONS.index(st.session_state["season"]),
    )
    if selected_season == "2024-25":
        st.markdown('<p style="color:#f5d060; font-size:0.75rem;">⚠️ Sezon devam ediyor</p>',
                    unsafe_allow_html=True)

    chart_mode_field = st.radio(
        "🗺️ Harita Modu",
        ["Noktasal", "Is Haritasi"],
        index=0 if st.session_state["chart_mode"] == "Noktasal" else 1,
    )
    num_games_field = st.slider(
        "🎮 Son Maç Sayısı",
        min_value=5, max_value=41,
        value=st.session_state["num_games"], step=1,
    )

    _changed = (
        selected_season     != st.session_state["season"]
        or chart_mode_field != st.session_state["chart_mode"]
        or num_games_field  != st.session_state["num_games"]
    )
    if _changed:
        if selected_season != st.session_state["season"]:
            fetch_game_log.clear(); fetch_shot_chart.clear(); find_player_id.clear()
        st.session_state.update({
            "season": selected_season,
            "chart_mode": chart_mode_field,
            "num_games": num_games_field,
        })
        st.rerun()

    st.divider()

    # Ollama durum gostergesi
    ollama_status = check_ollama_alive()
    if ollama_status:
        st.markdown(f"""
        <div style="background:rgba(61,220,132,0.1); border:1px solid #3ddc84;
                    border-radius:10px; padding:10px 14px; text-align:center;">
            <div style="color:#3ddc84; font-weight:700; font-size:0.85rem;">● OLLAMA AKTİF</div>
            <div style="color:#5a7a9a; font-size:0.72rem; margin-top:2px;">{OLLAMA_MODEL}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(255,107,107,0.1); border:1px solid #ff6b6b;
                    border-radius:10px; padding:10px 14px; text-align:center;">
            <div style="color:#ff6b6b; font-weight:700; font-size:0.85rem;">● OLLAMA KAPALI</div>
            <div style="color:#5a7a9a; font-size:0.72rem; margin-top:4px;">ollama serve</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#2a4a6b; font-size:0.68rem; text-align:center; line-height:1.8;">
        Veriler: NBA Stats API<br>
        Analiz: Yerel LLM (Offline)<br>
        🔒 Bulut bağımlılığı yok
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# SEKMELER
# ══════════════════════════════════════════════════════════
tab_single, tab_h2h = st.tabs([
    "👤  Tekli Oyuncu & AI Asistan",
    "⚔️  Head-to-Head Kıyaslama",
])

# ──────────────────────────────────────────────────────────
# SEKME 1 — TEKLI OYUNCU
# ──────────────────────────────────────────────────────────
with tab_single:

    # Arama formu
    with st.form("single_player_search_form"):
        col_s, col_b = st.columns([5, 1])
        with col_s:
            player_query_val = st.text_input(
                "Oyuncu Ara",
                value=st.session_state["active_player"],
                placeholder="🔍  Stephen Curry, LeBron James, Alperen Sengun...",
                label_visibility="collapsed",
            )
        with col_b:
            search_submitted = st.form_submit_button("🔍 Ara", use_container_width=True)

    if search_submitted and player_query_val.strip():
        st.session_state["active_player"] = player_query_val.strip()
        find_player_id.clear(); fetch_game_log.clear(); fetch_shot_chart.clear()
        st.rerun()

    # AI Router
    with st.expander("🤖  AI Router — Doğal Dille Soru Sor"):
        if not ollama_status:
            st.markdown("""
            <div style="background:rgba(255,107,107,0.1); border:1px solid #ff6b6b;
                        border-radius:10px; padding:12px 16px; color:#ff9999; font-size:0.87rem;">
                🔌 Ollama kapalı — AI Router devre dışı.<br>
                <code style="background:#1e2d45; padding:2px 8px; border-radius:4px; color:#f5d060;">ollama serve</code>
                komutunu çalıştırın.
            </div>
            """, unsafe_allow_html=True)
        with st.form("ai_router_form"):
            free_query = st.text_input(
                "Sorunuz:", label_visibility="collapsed",
                disabled=not ollama_status,
                placeholder="Ör: Luka Doncic son maçlarda nasıl oynuyor, şutları nasıl?",
            )
            ask_submitted = st.form_submit_button(
                "✨ Soruyu Yanıtla", type="primary",
                use_container_width=True, disabled=not ollama_status,
            )
        if ask_submitted and free_query.strip():
            with st.spinner("AI Router analiz ediyor..."):
                routed = route_user_query(free_query)
            detected = routed.get("player_name")
            clean_q  = routed.get("clean_question", free_query)
            if detected:
                st.session_state.update({"active_player": detected, "pending_question": clean_q})
                find_player_id.clear(); fetch_game_log.clear(); fetch_shot_chart.clear()
                st.rerun()
            else:
                st.session_state["pending_question"] = free_query
                st.info(f"Oyuncu adı bulunamadı → **{st.session_state['active_player']}** için yanıtlanacak.")

    # Oyuncu yukleme
    p_id, verified_name = find_player_id(st.session_state["active_player"])

    if p_id is None:
        st.markdown(f"""
        <div style="background:rgba(255,107,107,0.1); border:1px solid #ff6b6b;
                    border-radius:14px; padding:20px 24px; margin-top:20px;">
            <div style="font-size:1.5rem; margin-bottom:8px;">❌</div>
            <div style="color:#ff9999; font-weight:600;">"{st.session_state['active_player']}" bulunamadı.</div>
            <div style="color:#5a7a9a; font-size:0.85rem; margin-top:4px;">
                Lütfen İngilizce tam adını girin — ör: <em>LeBron James</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.pop("pending_question", None)
    else:
        with st.spinner(f"⏳ {verified_name} yükleniyor..."):
            g_log,   g_err = fetch_game_log(p_id,  st.session_state["season"])
            s_chart, s_err = fetch_shot_chart(p_id, st.session_state["season"])

        if g_err: st.toast(f"⚠️ Maç logu: {g_err}", icon="⚠️")
        if s_err: st.toast(f"⚠️ Şut haritası: {s_err}", icon="⚠️")

        if g_log.empty:
            st.warning(
                f"**{verified_name}** için **{st.session_state['season']}** sezonunda "
                "maç verisi bulunamadı. Farklı sezon seçin."
            )
            st.session_state.pop("pending_question", None)
        else:
            recent_g = g_log.tail(st.session_state["num_games"]).copy()
            z_stats  = compute_zone_stats(s_chart)

            # Hero Banner
            hero_banner(verified_name, st.session_state["season"], len(recent_g))

            # Metrik kartlari
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🏀  Sayı",          f"{recent_g['PTS'].mean():.1f}")
            c2.metric("📦  Ribaund",        f"{recent_g['REB'].mean():.1f}")
            c3.metric("🎯  Asist",          f"{recent_g['AST'].mean():.1f}")
            c4.metric("💯  Şut İsabeti",    f"{recent_g['FG_PCT'].mean()*100:.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            # Shot chart + zone stats
            col_shot, col_stats = st.columns([1.2, 1])
            with col_shot:
                section_header("🗺️", f"Şut Haritası — {st.session_state['chart_mode']}")
                fig = plot_shot_chart(
                    s_chart, title=verified_name,
                    mode=st.session_state["chart_mode"]
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with col_stats:
                section_header("📍", "Bölgesel Verimlilik")
                st.markdown(zone_table_html(z_stats), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                section_header("📈", "Sayı Trendi")
                bd = recent_g[["GAME_DATE", "PTS"]].copy()
                bd["GAME_DATE"] = bd["GAME_DATE"].dt.strftime("%d.%m")
                st.bar_chart(bd.set_index("GAME_DATE"), y="PTS", color="#c8a84b")

            st.markdown("<br>", unsafe_allow_html=True)

            # AI Analiz bolumu
            section_header("🧠", "Taktiksel AI Analizi")

            if not ollama_status:
                st.markdown("""
                <div style="background:rgba(255,107,107,0.08); border:1px solid #ff6b6b;
                            border-radius:12px; padding:14px 18px; color:#ff9999; font-size:0.87rem;">
                    🔌 AI analizi için Ollama çalışıyor olmalı.
                    Terminalde: <code style="background:#1e2d45; padding:2px 8px; border-radius:4px; color:#f5d060;">ollama serve</code>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            preset_qs = {
                "🛡️  Rakip Savunma Stratejisi": (
                    f"Rakip takım koçu olarak {verified_name}'e karşı nasıl savunma kurgulamalı? "
                    "Şut haritasındaki zayıf ve güçlü noktalara göre savunma planı çıkar."
                ),
                "🎯  Şut Tercihleri & Verimlilik": (
                    f"{verified_name}'in şut seçimi ne kadar verimli? "
                    "Boyalı alan, orta mesafe ve 3 sayı tercihlerini değerlendir."
                ),
                "📈  Form Trendi & Tutarlılık": (
                    f"{verified_name}'in son maçlarda skor, ribaund ve asist istikrarı nasıl? "
                    "Performansı yükseliyor mu?"
                ),
                "✍️  Özel Soru Yaz": "",
            }
            sel = st.selectbox("Analiz Konusu", list(preset_qs.keys()),
                               label_visibility="collapsed")
            default_txt = (
                preset_qs[sel] if sel != "✍️  Özel Soru Yaz"
                else f"{verified_name} hakkında taktiksel sorunuzu yazın..."
            )
            user_q = st.text_area("Soru", value=default_txt, height=80,
                                  label_visibility="collapsed")

            if st.button("🤖  Rapor Üret", type="primary",
                         disabled=not ollama_status, use_container_width=False):
                with st.spinner("Analiz hazırlanıyor..."):
                    st.session_state["last_ai_analysis"] = query_ollama_analysis(
                        verified_name, g_log, z_stats, user_q
                    )

            # pending_question guard
            if st.session_state.get("pending_question"):
                pending_q = st.session_state.pop("pending_question")
                with st.spinner("🤖 AI Router sorusuna yanıt hazırlanıyor..."):
                    st.session_state["last_ai_analysis"] = query_ollama_analysis(
                        verified_name, g_log, z_stats, pending_q
                    )

            if st.session_state["last_ai_analysis"]:
                styled_ai_output(st.session_state["last_ai_analysis"])


# ──────────────────────────────────────────────────────────
# SEKME 2 — HEAD-TO-HEAD
# ──────────────────────────────────────────────────────────
with tab_h2h:
    st.markdown(f"""
    <div class="nba-hero">
        <div style="font-size:2.8rem; margin-bottom:6px;">⚔️</div>
        <h1 style="margin:0; font-size:1.8rem;">Head-to-Head Karşılaştırma</h1>
        <div style="margin-top:10px; display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">
            <span class="badge badge-gold">📅 {st.session_state['season']}</span>
            <span class="badge badge-blue">🎮 Son {st.session_state['num_games']} Maç</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cp1, cp2, cbtn = st.columns([2, 2, 1])
    with cp1:
        p1_input = st.text_input("1. Oyuncu", value=st.session_state["h2h_p1"],
                                 placeholder="Ör: Alperen Sengun")
    with cp2:
        p2_input = st.text_input("2. Oyuncu", value=st.session_state["h2h_p2"],
                                 placeholder="Ör: Nikola Jokic")
    with cbtn:
        st.markdown("<br>", unsafe_allow_html=True)
        compare_btn = st.button("⚖️  Kıyasla", type="primary", use_container_width=True)

    if compare_btn:
        st.session_state.update({"h2h_p1": p1_input.strip(), "h2h_p2": p2_input.strip()})
        st.rerun()

    p1_id, p1_name = find_player_id(st.session_state["h2h_p1"])
    p2_id, p2_name = find_player_id(st.session_state["h2h_p2"])

    if not p1_id or not p2_id:
        missing = []
        if not p1_id: missing.append(st.session_state["h2h_p1"])
        if not p2_id: missing.append(st.session_state["h2h_p2"])
        st.error(f"❌ Bulunamadı: {' / '.join(missing)}. İngilizce tam adı girin.")
    else:
        with st.spinner("⏳ Veriler yükleniyor..."):
            p1_log,   p1_ge = fetch_game_log(p1_id,  st.session_state["season"])
            p1_shots, p1_se = fetch_shot_chart(p1_id, st.session_state["season"])
            p2_log,   p2_ge = fetch_game_log(p2_id,  st.session_state["season"])
            p2_shots, p2_se = fetch_shot_chart(p2_id, st.session_state["season"])

        for err in filter(None, [p1_ge, p1_se, p2_ge, p2_se]):
            st.toast(f"⚠️ {err}", icon="⚠️")

        if p1_log.empty or p2_log.empty:
            empties = []
            if p1_log.empty: empties.append(p1_name)
            if p2_log.empty: empties.append(p2_name)
            st.warning(f"⚠️ {' / '.join(empties)} için {st.session_state['season']} verisi yok.")
        else:
            n   = st.session_state["num_games"]
            p1r = p1_log.tail(n); p2r = p2_log.tail(n)

            p1_pts, p2_pts = p1r["PTS"].mean(), p2r["PTS"].mean()
            p1_reb, p2_reb = p1r["REB"].mean(), p2r["REB"].mean()
            p1_ast, p2_ast = p1r["AST"].mean(), p2r["AST"].mean()
            p1_fg,  p2_fg  = p1r["FG_PCT"].mean()*100, p2r["FG_PCT"].mean()*100

            def _ldr(v1, v2, n1, n2):
                return n1 if v1 >= v2 else n2

            # Kiyaslama tablosu — ozel HTML
            section_header("📊", f"İstatistiksel Kıyaslama — Son {n} Maç")

            rows_h2h = ""
            metrics_data = [
                ("🏀 Sayı",       p1_pts, p2_pts),
                ("📦 Ribaund",     p1_reb, p2_reb),
                ("🎯 Asist",       p1_ast, p2_ast),
                ("💯 FG%",         p1_fg,  p2_fg),
            ]
            for label, v1, v2 in metrics_data:
                leader    = p1_name if v1 >= v2 else p2_name
                diff      = abs(v1 - v2)
                bar1_w    = int(v1 / max(v1, v2) * 100)
                bar2_w    = int(v2 / max(v1, v2) * 100)
                c1_bold   = "font-weight:800; color:#f5d060;" if v1 >= v2 else "color:#8ba8c4;"
                c2_bold   = "font-weight:800; color:#f5d060;" if v2 >= v1 else "color:#8ba8c4;"
                diff_fmt  = f"%{diff:.1f}" if "FG" in label else f"{diff:.1f}"
                rows_h2h += f"""
                <tr style="border-bottom:1px solid #1e3a5f;">
                    <td style="padding:12px 14px; color:#8ba8c4; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.8px;">{label}</td>
                    <td style="padding:12px 14px; text-align:right; {c1_bold}">{v1:.1f}</td>
                    <td style="padding:12px 14px; text-align:center;">
                        <div style="display:flex; align-items:center; gap:4px; justify-content:center;">
                            <div style="width:{bar1_w//2}px; height:6px; background:{'#c8a84b' if v1>=v2 else '#2a4a6b'}; border-radius:3px; direction:rtl;"></div>
                            <span style="color:#2a4a6b; font-size:0.7rem;">vs</span>
                            <div style="width:{bar2_w//2}px; height:6px; background:{'#c8a84b' if v2>v1 else '#2a4a6b'}; border-radius:3px;"></div>
                        </div>
                        <div style="color:#3ddc84; font-size:0.72rem; margin-top:2px;">+{diff_fmt} → {leader}</div>
                    </td>
                    <td style="padding:12px 14px; text-align:left; {c2_bold}">{v2:.1f}</td>
                </tr>"""

            st.markdown(f"""
            <table style="width:100%; border-collapse:collapse; background:#111827;
                          border-radius:14px; overflow:hidden; border:1px solid #1e3a5f;">
                <thead>
                    <tr style="background:#1e2d45;">
                        <th style="padding:12px 14px; color:#5a7a9a; text-align:left; font-size:0.75rem; text-transform:uppercase;">Metrik</th>
                        <th style="padding:12px 14px; color:#c8a84b; text-align:right;">{p1_name}</th>
                        <th style="padding:12px 14px; color:#5a7a9a; text-align:center; font-size:0.75rem;">FARK</th>
                        <th style="padding:12px 14px; color:#c8a84b; text-align:left;">{p2_name}</th>
                    </tr>
                </thead>
                <tbody>{rows_h2h}</tbody>
            </table>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Shot haritalari
            section_header("🗺️", f"Şut Haritaları — {st.session_state['chart_mode']}")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown(f"<p style='text-align:center; color:#c8a84b; font-weight:700; margin-bottom:4px;'>{p1_name}</p>", unsafe_allow_html=True)
                f1 = plot_shot_chart(p1_shots, title="", mode=st.session_state["chart_mode"])
                st.pyplot(f1, use_container_width=True); plt.close(f1)
            with cc2:
                st.markdown(f"<p style='text-align:center; color:#c8a84b; font-weight:700; margin-bottom:4px;'>{p2_name}</p>", unsafe_allow_html=True)
                f2 = plot_shot_chart(p2_shots, title="", mode=st.session_state["chart_mode"])
                st.pyplot(f2, use_container_width=True); plt.close(f2)

            # Zone karsilastirma
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("📍", "Bölgesel İsabet Karşılaştırması")
            p1z = compute_zone_stats(p1_shots)
            p2z = compute_zone_stats(p2_shots)
            zc1, zc2 = st.columns(2)
            with zc1:
                st.markdown(f"<p style='color:#c8a84b; font-weight:700; margin-bottom:6px;'>{p1_name}</p>", unsafe_allow_html=True)
                st.markdown(zone_table_html(p1z), unsafe_allow_html=True)
            with zc2:
                st.markdown(f"<p style='color:#c8a84b; font-weight:700; margin-bottom:6px;'>{p2_name}</p>", unsafe_allow_html=True)
                st.markdown(zone_table_html(p2z), unsafe_allow_html=True)

            # AI Rapor
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("🧠", "Yerel AI Scouting Raporu")

            if not ollama_status:
                st.markdown("""
                <div style="background:rgba(255,107,107,0.08); border:1px solid #ff6b6b;
                            border-radius:12px; padding:14px 18px; color:#ff9999; font-size:0.87rem;">
                    🔌 AI raporu için Ollama çalışıyor olmalı.
                    Terminalde: <code style="background:#1e2d45; padding:2px 8px; border-radius:4px; color:#f5d060;">ollama serve</code>
                </div>
                """, unsafe_allow_html=True)

            if st.button("🤖  Çapraz Scouting Raporu Üret", type="primary",
                         disabled=not ollama_status):
                with st.spinner("Rapor hazırlanıyor..."):
                    def _zstr(zdf):
                        return (", ".join(f"{r['Bolge']}: %{r['Yuzde']}" for _, r in zdf.iterrows())
                                if not zdf.empty else "Veri yok")
                    p1_payload = {"pts": p1_pts, "reb": p1_reb, "ast": p1_ast,
                                  "fg": p1_fg, "zones": _zstr(p1z)}
                    p2_payload = {"pts": p2_pts, "reb": p2_reb, "ast": p2_ast,
                                  "fg": p2_fg, "zones": _zstr(p2z)}
                    st.session_state["last_h2h_analysis"] = query_ollama_comparison(
                        p1_name, p1_payload, p2_name, p2_payload
                    )

            if st.session_state["last_h2h_analysis"]:
                styled_ai_output(st.session_state["last_h2h_analysis"])