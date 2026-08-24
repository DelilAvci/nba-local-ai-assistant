# -*- coding: utf-8 -*-
"""
NBA Game & Player Assistant Agent  (v2)
---------------------------------------
Gelistirmeler:
  1. Dinamik sezon secimi (2021-22 -> 2024-25)
  2. Guclu API hata yonetimi: 2x retry + st.toast bildirimleri
  3. AI Router duzeltmesi: pending_question guard,
     3-adimli JSON parse, guclendirilmis system prompt,
     Ollama kapali uyarisi, ayri st.form

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

# ──────────────────────────────────────────────────────────
# GENEL AYARLAR
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Local AI Assistant",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

OLLAMA_CHAT_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL      = "qwen2.5:7b"
REQUEST_TIMEOUT   = 120
AVAILABLE_SEASONS = ["2024-25", "2023-24", "2022-23", "2021-22"]

# ──────────────────────────────────────────────────────────
# VERI CEKME — CACHE + RETRY
# ──────────────────────────────────────────────────────────

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
    """Mac loglarini ceker. Returns (DataFrame, error_str | None). Max 2 retry."""
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
            last_error = "NBA API baglantisi kurulamadi. Internet baglantinizi kontrol edin."
        except Exception as exc:
            last_error = f"Mac logu cekilemedi: {exc}"
        if attempt < 2:
            time.sleep(1.5)
    return pd.DataFrame(), last_error


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_shot_chart(player_id: int, season: str):
    """Sut haritasi verisini ceker. Returns (DataFrame, error_str | None). Max 2 retry."""
    last_error = None
    for attempt in range(1, 3):
        try:
            sc = shotchartdetail.ShotChartDetail(
                team_id=0,
                player_id=player_id,
                season_nullable=season,
                season_type_all_star="Regular Season",
                context_measure_simple="FGA",
                timeout=90,
            )
            return sc.get_data_frames()[0], None
        except requests.exceptions.Timeout:
            last_error = f"Sut haritasi zaman asimi (deneme {attempt}/2)."
        except requests.exceptions.ConnectionError:
            last_error = "NBA API baglantisi kurulamadi."
        except Exception as exc:
            last_error = f"Sut haritasi cekilemedi: {exc}"
        if attempt < 2:
            time.sleep(1.5)
    return pd.DataFrame(), last_error


# ──────────────────────────────────────────────────────────
# SAHA CIZIMI & ISTATISTIK
# ──────────────────────────────────────────────────────────

def draw_court(ax=None, color="white", lw=1.6):
    if ax is None:
        ax = plt.gca()
    for el in [
        patches.Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False),
        patches.Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color),
        patches.Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False),
        patches.Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False),
        patches.Arc((0, 142.5), 120, 120, theta1=0,   theta2=180, linewidth=lw, color=color),
        patches.Arc((0, 142.5), 120, 120, theta1=180, theta2=360, linewidth=lw, color=color, linestyle="dashed"),
        patches.Arc((0, 0),      80,  80, theta1=0,   theta2=180, linewidth=lw, color=color),
        patches.Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color),
        patches.Rectangle(( 220, -47.5), 0, 140, linewidth=lw, color=color),
        patches.Arc((0, 0),     475, 475, theta1=22,  theta2=158, linewidth=lw, color=color),
        patches.Arc((0, 422.5), 120, 120, theta1=180, theta2=0,   linewidth=lw, color=color),
        patches.Rectangle((-250, -47.5), 500, 470, linewidth=lw, color=color, fill=False),
    ]:
        ax.add_patch(el)
    ax.set_xlim(-260, 260)
    ax.set_ylim(-60, 430)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#0b1d0f")
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def plot_shot_chart(shot_df, title="", mode="Noktasal"):
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    fig.patch.set_facecolor("#0e1117")
    draw_court(ax)
    if shot_df.empty:
        ax.text(0, 200, "Sut verisi bulunamadi", color="white",
                ha="center", va="center", fontsize=11)
        ax.set_title(title or "Sut Dagilim Haritasi", color="white", fontsize=12, pad=10)
        fig.tight_layout()
        return fig
    if mode == "Is Haritasi":
        hb = ax.hexbin(shot_df["LOC_X"], shot_df["LOC_Y"],
                       gridsize=26, extent=(-260, 260, -60, 430),
                       cmap="inferno", mincnt=1, alpha=0.85)
        cbar = fig.colorbar(hb, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Sut Yogunlugu", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax, "yticklabels"), color="white")
    else:
        made   = shot_df[shot_df["SHOT_MADE_FLAG"] == 1]
        missed = shot_df[shot_df["SHOT_MADE_FLAG"] == 0]
        ax.scatter(missed["LOC_X"], missed["LOC_Y"],
                   c="#ff4d4d", marker="x", s=30, alpha=0.75,
                   label=f"Kacan ({len(missed)})")
        ax.scatter(made["LOC_X"], made["LOC_Y"],
                   c="#3ddc84", marker="o", s=30, alpha=0.85,
                   edgecolors="white", linewidths=0.3,
                   label=f"Isabetli ({len(made)})")
        ax.legend(loc="upper right", facecolor="#1c1f26",
                  labelcolor="white", fontsize=8.5, framealpha=0.9)
    ax.set_title(title or "Sut Dagilim Haritasi", color="white", fontsize=12, pad=10)
    fig.tight_layout()
    return fig


def compute_zone_stats(shot_df):
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


# ──────────────────────────────────────────────────────────
# OLLAMA AI
# ──────────────────────────────────────────────────────────

def check_ollama_alive() -> bool:
    try:
        return requests.get("http://localhost:11434/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


def _parse_json_safe(text: str):
    """3-adimli guvenli JSON parse: dogrudan -> brace-regex -> markdown blogu."""
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
        "Your ONLY job is to extract two things from the user input:\n"
        "  1. The full NBA player name.\n"
        "  2. The specific question about that player.\n\n"
        "Return ONLY a valid JSON object:\n"
        '  {"player_name": "Full Player Name", "clean_question": "..."}\n\n'
        "Rules:\n"
        "- Use the player full name (first + last).\n"
        "- If no player is mentioned, set player_name to null.\n"
        "- No extra text outside the JSON.\n\n"
        "Examples:\n"
        'Input: "LeBron son maclarda nasil oynadı?"\n'
        'Output: {"player_name": "LeBron James", "clean_question": "Son maclardaki performansi nasil?"}\n'
        'Input: "Alperen Sengun boyali alandaki sut yuzdesi nedir?"\n'
        'Output: {"player_name": "Alperen Sengun", "clean_question": "Boyali alandaki sut yuzdesi?"}\n'
        'Input: "Curry uc sayi vuruyor mu?"\n'
        'Output: {"player_name": "Stephen Curry", "clean_question": "Uc sayi isabeti nasil?"}'
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
        st.toast("AI Router zaman asimina ugradi, soru dogrudan kullaniliyor.", icon="⚠️")
    except requests.exceptions.ConnectionError:
        st.toast("Ollama baglantisi kurulamadi.", icon="🔴")
    except Exception as exc:
        st.toast(f"AI Router hatasi: {exc}", icon="⚠️")
    return {"player_name": None, "clean_question": user_text}


def query_ollama_analysis(player_name: str, game_log, zone_stats, user_question: str) -> str:
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
        "Her baslik 2-3 cumle. Tekrara dusme, donguye girme."
    )
    user_prompt = (
        f"Oyuncu: {player_name}\n\nSon 5 Mac:\n" + "\n".join(lines) +
        "\n\nBolgesel Sut %:\n" + "\n".join(zone_lines) +
        f"\n\nKullanici Sorusu: {user_question}"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.85, "num_predict": 350},
    }
    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except requests.exceptions.Timeout:
        return f"AI zaman asimina ugradi ({REQUEST_TIMEOUT}s). Tekrar deneyin."
    except requests.exceptions.ConnectionError:
        return "Ollama kapali veya erisilemez (localhost:11434)."
    except Exception as exc:
        return f"AI Baglanti Hatasi: {exc}"


def query_ollama_comparison(p1_name: str, p1: dict, p2_name: str, p2: dict) -> str:
    system_prompt = (
        "Sen profesyonel bir NBA scout ve analistisin. Iki oyuncuyu kiyaslayan "
        "kisa, net ve akici TURKCE scouting raporu yaz. 3 baslik: "
        "1) Skor & Verimlilik Karsilastirmasi  "
        "2) Sut Haritasi & Alan Hakimiyeti  "
        "3) Koc Esleme Notu (Hangisi Avantajli?). "
        "Her baslik 2-3 cumle. Tekrara dusme."
    )
    user_prompt = (
        f"OYUNCU 1: {p1_name}\n"
        f"Ortalamalar: {p1['pts']:.1f} Sayi, {p1['reb']:.1f} Rib, {p1['ast']:.1f} Ast, %{p1['fg']:.1f} FG\n"
        f"Sut Bolgeleri: {p1['zones']}\n\n"
        f"OYUNCU 2: {p2_name}\n"
        f"Ortalamalar: {p2['pts']:.1f} Sayi, {p2['reb']:.1f} Rib, {p2['ast']:.1f} Ast, %{p2['fg']:.1f} FG\n"
        f"Sut Bolgeleri: {p2['zones']}\n\n"
        "Taktiksel karsilastirma:"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.85, "num_predict": 400},
    }
    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except requests.exceptions.Timeout:
        return f"AI zaman asimina ugradi ({REQUEST_TIMEOUT}s). Tekrar deneyin."
    except requests.exceptions.ConnectionError:
        return "Ollama kapali veya erisilemez (localhost:11434)."
    except Exception as exc:
        return f"AI Baglanti Hatasi: {exc}"


# ──────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────
st.sidebar.title("🏀 NBA Local AI Assistant")
st.sidebar.caption(f"Yerel LLM: {OLLAMA_MODEL}")

selected_season = st.sidebar.selectbox(
    "🗓️ Sezon Secin:",
    options=AVAILABLE_SEASONS,
    index=AVAILABLE_SEASONS.index(st.session_state["season"]),
)
if selected_season == "2024-25":
    st.sidebar.caption("⚠️ 2024-25 sezonu devam ediyor — bazi oyuncularin verisi eksik olabilir.")

chart_mode_field = st.sidebar.radio(
    "Sut Haritasi Gorunumu:",
    ["Noktasal", "Is Haritasi"],
    index=0 if st.session_state["chart_mode"] == "Noktasal" else 1,
)
num_games_field = st.sidebar.slider(
    "Son Mac Sayisi:", min_value=5, max_value=41,
    value=st.session_state["num_games"], step=1,
)

_changed = (
    selected_season     != st.session_state["season"]
    or chart_mode_field != st.session_state["chart_mode"]
    or num_games_field  != st.session_state["num_games"]
)
if _changed:
    if selected_season != st.session_state["season"]:
        fetch_game_log.clear()
        fetch_shot_chart.clear()
        find_player_id.clear()
    st.session_state.update({
        "season":     selected_season,
        "chart_mode": chart_mode_field,
        "num_games":  num_games_field,
    })
    st.rerun()

st.sidebar.divider()
ollama_status = check_ollama_alive()
if ollama_status:
    st.sidebar.success(f"✅ Ollama Aktif ({OLLAMA_MODEL})")
else:
    st.sidebar.error("🔴 Ollama Kapali (localhost:11434)")
    st.sidebar.caption("AI ozellikleri devre disi. Terminalde: ollama serve")

# ──────────────────────────────────────────────────────────
# SEKMELER
# ──────────────────────────────────────────────────────────
tab_single, tab_h2h = st.tabs([
    "👤 Tekli Oyuncu & Akilli Asistan",
    "⚔️ Head-to-Head Kiyaslama",
])

# ══════════════════════════════════════════════════════════
# SEKME 1 — TEKLI OYUNCU
# ══════════════════════════════════════════════════════════
with tab_single:
    st.title("🏀 Oyuncu Analiz & Taktik Paneli")
    st.caption(f"📅 Aktif Sezon: **{st.session_state['season']}**")

    # Dogrudan arama formu
    with st.form("single_player_search_form"):
        col_s, col_b = st.columns([4, 1])
        with col_s:
            player_query_val = st.text_input(
                "🔍 Oyuncu Adi Girin:",
                value=st.session_state["active_player"],
                placeholder="Orn: Stephen Curry, LeBron James, Alperen Sengun...",
            )
        with col_b:
            st.write("")
            st.write("")
            search_submitted = st.form_submit_button("Oyuncuyu Getir", use_container_width=True)

    if search_submitted and player_query_val.strip():
        st.session_state["active_player"] = player_query_val.strip()
        find_player_id.clear()
        fetch_game_log.clear()
        fetch_shot_chart.clear()
        st.rerun()

    # AI Router — ayri form (lagging-state duzeltmesi)
    with st.expander("🤖 Dogal Dille Soru Sor (AI Router)"):
        if not ollama_status:
            st.warning(
                "Ollama kapali — AI Router devre disi. Terminalde: ollama serve",
                icon="🔌",
            )
        with st.form("ai_router_form"):
            free_query = st.text_input(
                "Sorunuz:",
                label_visibility="collapsed",
                disabled=not ollama_status,
                placeholder="Orn: Luka Doncic son maclarda nasil oynadı?",
            )
            ask_submitted = st.form_submit_button(
                "🔍 Soruyu Yanitla",
                type="primary",
                use_container_width=True,
                disabled=not ollama_status,
            )
        if ask_submitted and free_query.strip():
            with st.spinner("AI Router soruyu inceliyor..."):
                routed = route_user_query(free_query)
            detected = routed.get("player_name")
            clean_q  = routed.get("clean_question", free_query)
            if detected:
                st.session_state.update({
                    "active_player":    detected,
                    "pending_question": clean_q,
                })
                find_player_id.clear()
                fetch_game_log.clear()
                fetch_shot_chart.clear()
                st.rerun()
            else:
                st.session_state["pending_question"] = free_query
                st.info(
                    f"Soru icinde oyuncu adi bulunamadi. "
                    f"Mevcut oyuncu icin yanitlanacak: **{st.session_state['active_player']}**"
                )

    # Oyuncu yukleme
    p_id, verified_name = find_player_id(st.session_state["active_player"])

    if p_id is None:
        st.error(
            f"❌ **'{st.session_state['active_player']}'** bulunamadi. "
            "Lutfen Ingilizce tam adini girin (orn: 'LeBron James')."
        )
        st.session_state.pop("pending_question", None)
    else:
        with st.spinner(f"⏳ {verified_name} verileri yukleniyor..."):
            g_log,   g_err = fetch_game_log(p_id,  st.session_state["season"])
            s_chart, s_err = fetch_shot_chart(p_id, st.session_state["season"])

        if g_err:
            st.toast(f"⚠️ Mac logu: {g_err}", icon="⚠️")
        if s_err:
            st.toast(f"⚠️ Sut haritasi: {s_err}", icon="⚠️")

        if g_log.empty:
            st.warning(
                f"⚠️ **{verified_name}** icin **{st.session_state['season']}** sezonunda "
                "mac verisi bulunamadi. Farkli bir sezon secin."
            )
            st.session_state.pop("pending_question", None)
        else:
            recent_g = g_log.tail(st.session_state["num_games"]).copy()
            z_stats  = compute_zone_stats(s_chart)

            st.divider()
            st.subheader(f"📊 {verified_name} — Son {len(recent_g)} Mac Ortalamalari")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sayi (PTS)",        f"{recent_g['PTS'].mean():.1f}")
            c2.metric("Ribaund (REB)",      f"{recent_g['REB'].mean():.1f}")
            c3.metric("Asist (AST)",        f"{recent_g['AST'].mean():.1f}")
            c4.metric("Sut Isabeti (FG%)",  f"{recent_g['FG_PCT'].mean()*100:.1f}%")

            st.divider()
            col_shot, col_stats = st.columns([1.15, 1])
            with col_shot:
                st.subheader(f"🎯 Sut Haritasi ({st.session_state['chart_mode']})")
                fig = plot_shot_chart(
                    s_chart, title=verified_name, mode=st.session_state["chart_mode"]
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with col_stats:
                st.subheader("📍 Bolgesel Verimlilik")
                if not z_stats.empty:
                    st.dataframe(z_stats, use_container_width=True, hide_index=True)
                else:
                    st.info("Sut bolgesi verisi yok.")
                st.markdown("**Son Maclarda Sayi Trendi**")
                bd = recent_g[["GAME_DATE", "PTS"]].copy()
                bd["GAME_DATE"] = bd["GAME_DATE"].dt.strftime("%d.%m")
                st.bar_chart(bd.set_index("GAME_DATE"), y="PTS", color="#3ddc84")

            st.divider()
            st.subheader("🧠 Taktiksel AI Analizi")
            if not ollama_status:
                st.warning(
                    "AI analizi icin Ollama calisiyor olmali. Terminalde: ollama serve",
                    icon="🔌",
                )

            preset_qs = {
                "🛡️ Rakip Savunma Stratejisi": (
                    f"Rakip takim kocu olarak {verified_name} e karsi nasil savunma kurmali? "
                    "Sut haritasindaki zayif ve guclu noktalara gore plan cikar."
                ),
                "🎯 Sut Tercihleri ve Verimlilik": (
                    f"{verified_name} in sut secimi ne kadar verimli? "
                    "Boyali alan, orta mesafe ve 3 sayi tercihlerini degerlendir."
                ),
                "📈 Form Trendi ve Tutarlilik": (
                    f"{verified_name} in son maclarda skor, ribaund ve asist istikrari nasil? "
                    "Performansi yukseliyor mu?"
                ),
                "✍️ Ozel Soru Yaz": "",
            }
            sel = st.selectbox("📌 Analiz Konusu Secin:", list(preset_qs.keys()))
            default_txt = (
                preset_qs[sel] if sel != "✍️ Ozel Soru Yaz"
                else f"{verified_name} hakkinda taktiksel soru..."
            )
            user_q = st.text_area("Taktiksel Soru:", value=default_txt, height=80)

            if st.button("🤖 Rapor Uret", type="primary", disabled=not ollama_status):
                with st.spinner("🔍 Analiz hazirlaniyor..."):
                    st.session_state["last_ai_analysis"] = query_ollama_analysis(
                        verified_name, g_log, z_stats, user_q
                    )

            # pending_question guard — sadece oyuncu basariyla yuklendikten sonra tetikle
            if st.session_state.get("pending_question"):
                pending_q = st.session_state.pop("pending_question")
                with st.spinner("🤖 AI Router sorusuna yanit hazirlaniyor..."):
                    st.session_state["last_ai_analysis"] = query_ollama_analysis(
                        verified_name, g_log, z_stats, pending_q
                    )

            if st.session_state["last_ai_analysis"]:
                st.info(st.session_state["last_ai_analysis"])


# ══════════════════════════════════════════════════════════
# SEKME 2 — HEAD-TO-HEAD
# ══════════════════════════════════════════════════════════
with tab_h2h:
    st.title("⚔️ NBA Head-to-Head Karsilastirma")
    st.caption(f"📅 Aktif Sezon: **{st.session_state['season']}**")
    st.markdown(
        "Iki oyuncunun sut haritalari, bolge hakimiyeti ve "
        "mac istatistiklerini yan yana kiyaslayin."
    )

    cp1, cp2, cbtn = st.columns([2, 2, 1])
    with cp1:
        p1_input = st.text_input("1. Oyuncu:", value=st.session_state["h2h_p1"])
    with cp2:
        p2_input = st.text_input("2. Oyuncu:", value=st.session_state["h2h_p2"])
    with cbtn:
        st.write("")
        st.write("")
        compare_btn = st.button("⚖️ Kiyasla", type="primary", use_container_width=True)

    if compare_btn:
        st.session_state.update({
            "h2h_p1": p1_input.strip(),
            "h2h_p2": p2_input.strip(),
        })
        st.rerun()

    p1_id, p1_name = find_player_id(st.session_state["h2h_p1"])
    p2_id, p2_name = find_player_id(st.session_state["h2h_p2"])

    if not p1_id or not p2_id:
        missing = []
        if not p1_id:
            missing.append(f"**{st.session_state['h2h_p1']}**")
        if not p2_id:
            missing.append(f"**{st.session_state['h2h_p2']}**")
        st.error(f"❌ Bulunamadi: {' ve '.join(missing)}. Ingilizce tam adi girin.")
    else:
        with st.spinner("⏳ Iki oyuncunun verileri yukleniyor..."):
            p1_log,   p1_ge = fetch_game_log(p1_id,  st.session_state["season"])
            p1_shots, p1_se = fetch_shot_chart(p1_id, st.session_state["season"])
            p2_log,   p2_ge = fetch_game_log(p2_id,  st.session_state["season"])
            p2_shots, p2_se = fetch_shot_chart(p2_id, st.session_state["season"])

        for err in filter(None, [p1_ge, p1_se, p2_ge, p2_se]):
            st.toast(f"⚠️ {err}", icon="⚠️")

        if p1_log.empty or p2_log.empty:
            empties = []
            if p1_log.empty:
                empties.append(p1_name)
            if p2_log.empty:
                empties.append(p2_name)
            st.warning(
                f"⚠️ {' ve '.join(empties)} icin {st.session_state['season']} "
                "sezonunda veri bulunamadi. Farkli sezon secin."
            )
        else:
            n   = st.session_state["num_games"]
            p1r = p1_log.tail(n)
            p2r = p2_log.tail(n)

            p1_pts, p2_pts = p1r["PTS"].mean(),        p2r["PTS"].mean()
            p1_reb, p2_reb = p1r["REB"].mean(),        p2r["REB"].mean()
            p1_ast, p2_ast = p1r["AST"].mean(),        p2r["AST"].mean()
            p1_fg,  p2_fg  = p1r["FG_PCT"].mean()*100, p2r["FG_PCT"].mean()*100

            def _ldr(v1, v2, n1, n2):
                return n1 if v1 >= v2 else n2

            st.subheader(f"📊 Istatistiksel Kiyaslama — Son {n} Mac")
            mdf = pd.DataFrame({
                "Metrik": ["Sayi", "Ribaund", "Asist", "FG%"],
                p1_name:  [f"{p1_pts:.1f}", f"{p1_reb:.1f}", f"{p1_ast:.1f}", f"%{p1_fg:.1f}"],
                p2_name:  [f"{p2_pts:.1f}", f"{p2_reb:.1f}", f"{p2_ast:.1f}", f"%{p2_fg:.1f}"],
                "Lider":  [
                    f"{abs(p1_pts-p2_pts):.1f} -> {_ldr(p1_pts,p2_pts,p1_name,p2_name)}",
                    f"{abs(p1_reb-p2_reb):.1f} -> {_ldr(p1_reb,p2_reb,p1_name,p2_name)}",
                    f"{abs(p1_ast-p2_ast):.1f} -> {_ldr(p1_ast,p2_ast,p1_name,p2_name)}",
                    f"%{abs(p1_fg-p2_fg):.1f} -> {_ldr(p1_fg,p2_fg,p1_name,p2_name)}",
                ],
            })
            st.dataframe(mdf, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader(f"🎯 Sut Haritalari ({st.session_state['chart_mode']})")
            cc1, cc2 = st.columns(2)
            with cc1:
                f1 = plot_shot_chart(p1_shots, title=p1_name, mode=st.session_state["chart_mode"])
                st.pyplot(f1, use_container_width=True)
                plt.close(f1)
            with cc2:
                f2 = plot_shot_chart(p2_shots, title=p2_name, mode=st.session_state["chart_mode"])
                st.pyplot(f2, use_container_width=True)
                plt.close(f2)

            st.subheader("📍 Bolgesel Isabet Karsilastirmasi")
            # Orijinal DataFrame'ler (p1z, p2z) AI zone string icin korunuyor
            p1z = compute_zone_stats(p1_shots)
            p2z = compute_zone_stats(p2_shots)
            # Gosterim icin rename
            p1zd = p1z.rename(columns={
                "Yuzde":    f"{p1_name} %",
                "Denenen":  f"{p1_name} Deneme",
                "Isabetli": f"{p1_name} Isabetli",
            })
            p2zd = p2z.rename(columns={
                "Yuzde":    f"{p2_name} %",
                "Denenen":  f"{p2_name} Deneme",
                "Isabetli": f"{p2_name} Isabetli",
            })
            merged = pd.merge(p1zd, p2zd, on="Bolge", how="outer").fillna("-")
            st.dataframe(merged, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("🧠 Yerel AI Head-to-Head Scouting Raporu")
            if not ollama_status:
                st.warning(
                    "AI raporu icin Ollama calisiyor olmali. Terminalde: ollama serve",
                    icon="🔌",
                )

            if st.button(
                "🤖 Iki Oyuncuyu Kiyasla — Rapor Uret",
                type="primary",
                disabled=not ollama_status,
            ):
                with st.spinner("🔍 Capraz scouting raporu hazirlaniyor..."):
                    def _zstr(zdf):
                        return (
                            ", ".join(f"{r['Bolge']}: %{r['Yuzde']}" for _, r in zdf.iterrows())
                            if not zdf.empty else "Veri yok"
                        )
                    p1_payload = {
                        "pts": p1_pts, "reb": p1_reb,
                        "ast": p1_ast, "fg":  p1_fg,
                        "zones": _zstr(p1z),
                    }
                    p2_payload = {
                        "pts": p2_pts, "reb": p2_reb,
                        "ast": p2_ast, "fg":  p2_fg,
                        "zones": _zstr(p2z),
                    }
                    st.session_state["last_h2h_analysis"] = query_ollama_comparison(
                        p1_name, p1_payload, p2_name, p2_payload
                    )

            if st.session_state["last_h2h_analysis"]:
                st.info(st.session_state["last_h2h_analysis"])