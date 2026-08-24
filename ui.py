# -*- coding: utf-8 -*-
"""
NBA Game & Player Assistant Agent (Single + Head-to-Head Comparison)
---------------------------------------------------------------------
- Veri: nba_api (PlayerGameLog, ShotChartDetail)
- Görselleştirme: Streamlit + Matplotlib
- Yerel AI: Ollama Chat API (phi3:mini) -> Router & Head-to-Head Analyst
"""

import json
import re
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from nba_api.stats.endpoints import playergamelog, shotchartdetail
from nba_api.stats.static import players

# ============================================================
# GENEL AYARLAR
# ============================================================
st.set_page_config(
    page_title="NBA Local AI Assistant",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "phi3:mini"
REQUEST_TIMEOUT = 120
CURRENT_SEASON = "2023-24"

# ============================================================
# VERİ ÇEKME FONKSİYONLARI (CACHE'Lİ)
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def find_player_id(player_name: str):
    if not player_name or not player_name.strip():
        return None, None
    matches = players.find_players_by_full_name(player_name.strip())
    if not matches:
        return None, None
    active = [p for p in matches if p.get("is_active")]
    target = active[0] if active else matches[0]
    return target["id"], target["full_name"]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_game_log(player_id: int, season: str) -> pd.DataFrame:
    try:
        log = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=60)
        df = log.get_data_frames()[0]
        if df.empty:
            return df
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_shot_chart(player_id: int, season: str) -> pd.DataFrame:
    try:
        sc = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=player_id,
            season_nullable=season,
            season_type_all_star="Regular Season",
            context_measure_simple="FGA",
            timeout=60,
        )
        return sc.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()


# ============================================================
# SAHA ÇİZİMİ & İSTATİSTİK
# ============================================================

def draw_court(ax=None, color="white", lw=1.6):
    if ax is None:
        ax = plt.gca()

    hoop = patches.Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
    backboard = patches.Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)
    outer_box = patches.Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
    inner_box = patches.Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)
    top_free_throw = patches.Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=lw, color=color)
    bottom_free_throw = patches.Arc((0, 142.5), 120, 120, theta1=180, theta2=360, linewidth=lw, color=color, linestyle="dashed")
    restricted = patches.Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)
    corner_three_left = patches.Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color)
    corner_three_right = patches.Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    three_arc = patches.Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)
    center_outer_arc = patches.Arc((0, 422.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color)
    outer = patches.Rectangle((-250, -47.5), 500, 470, linewidth=lw, color=color, fill=False)

    for el in [hoop, backboard, outer_box, inner_box, top_free_throw, bottom_free_throw,
               restricted, corner_three_left, corner_three_right, three_arc, center_outer_arc, outer]:
        ax.add_patch(el)

    ax.set_xlim(-260, 260)
    ax.set_ylim(-60, 430)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#0b1d0f")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def plot_shot_chart(shot_df: pd.DataFrame, title: str = "", mode: str = "Noktasal"):
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    fig.patch.set_facecolor("#0e1117")
    draw_court(ax, color="white", lw=1.6)

    if shot_df.empty:
        ax.text(0, 200, "Şut verisi bulunamadı", color="white", ha="center", va="center", fontsize=11)
        return fig

    made = shot_df[shot_df["SHOT_MADE_FLAG"] == 1]
    missed = shot_df[shot_df["SHOT_MADE_FLAG"] == 0]

    if mode == "Isı Haritası":
        hb = ax.hexbin(shot_df["LOC_X"], shot_df["LOC_Y"], gridsize=26, extent=(-260, 260, -60, 430), cmap="inferno", mincnt=1, alpha=0.85)
        cbar = fig.colorbar(hb, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Şut Yoğunluğu", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax, "yticklabels"), color="white")
    else:
        ax.scatter(missed["LOC_X"], missed["LOC_Y"], c="#ff4d4d", marker="x", s=30, alpha=0.75, label=f"Kaçan ({len(missed)})")
        ax.scatter(made["LOC_X"], made["LOC_Y"], c="#3ddc84", marker="o", s=30, alpha=0.85, edgecolors="white", linewidths=0.3, label=f"İsabetli ({len(made)})")
        ax.legend(loc="upper right", facecolor="#1c1f26", labelcolor="white", fontsize=8.5, framealpha=0.9)

    ax.set_title(title or "Şut Dağılım Haritası", color="white", fontsize=12, pad=10)
    fig.tight_layout()
    return fig


def compute_zone_stats(shot_df: pd.DataFrame) -> pd.DataFrame:
    if shot_df.empty:
        return pd.DataFrame(columns=["Bölge", "Denenen", "İsabetli", "Yüzde"])

    zone_map = {
        "Restricted Area": "Boyalı Alan",
        "In The Paint (Non-RA)": "Boyalı Alan",
        "Mid-Range": "Orta Mesafe",
        "Left Corner 3": "3 Sayı",
        "Right Corner 3": "3 Sayı",
        "Above the Break 3": "3 Sayı",
        "Backcourt": "Diğer",
    }
    df = shot_df.copy()
    df["ZONE_TR"] = df["SHOT_ZONE_BASIC"].map(zone_map).fillna("Diğer")

    grouped = df.groupby("ZONE_TR").agg(
        Denenen=("SHOT_MADE_FLAG", "count"),
        Isabetli=("SHOT_MADE_FLAG", "sum"),
    ).reset_index()
    grouped["Yüzde"] = (grouped["Isabetli"] / grouped["Denenen"] * 100).round(1)
    grouped = grouped.rename(columns={"ZONE_TR": "Bölge", "Isabetli": "İsabetli"})
    order = {"Boyalı Alan": 0, "Orta Mesafe": 1, "3 Sayı": 2, "Diğer": 3}
    grouped["_order"] = grouped["Bölge"].map(order).fillna(9)
    return grouped.sort_values("_order").drop(columns="_order").reset_index(drop=True)


# ============================================================
# OLLAMA AI ENTEGRASYONU
# ============================================================

def check_ollama_alive() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def route_user_query(user_text: str) -> dict:
    system_prompt = (
        "You are an NLP router for an NBA Assistant. Extract the NBA player's full name "
        "and the specific user question from the input. "
        "Return ONLY a valid JSON object: "
        '{"player_name": "Full Player Name", "clean_question": "Extracted question"}. '
        "If no player is mentioned, set player_name to null."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract info from: {user_text}"}
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 100}
    }

    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=20)
        resp.raise_for_status()
        raw_content = resp.json().get("message", {}).get("content", "").strip()
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return {"player_name": None, "clean_question": user_text}


def query_ollama_analysis(player_name: str, game_log: pd.DataFrame, zone_stats: pd.DataFrame, user_question: str) -> str:
    recent = game_log.tail(5)
    recent_lines = []
    for _, row in recent.iterrows():
        d_str = row['GAME_DATE'].strftime('%d.%m') if hasattr(row['GAME_DATE'], 'strftime') else str(row['GAME_DATE'])
        recent_lines.append(f"- Tarih {d_str} ({row['MATCHUP']}): {row['PTS']} Sayı, {row['REB']} Rib, {row['AST']} Ast, FG %{row['FG_PCT']*100:.0f}")
    recent_str = "\n".join(recent_lines)

    zone_lines = []
    if not zone_stats.empty:
        for _, row in zone_stats.iterrows():
            zone_lines.append(f"- {row['Bölge']}: {row['İsabetli']}/{row['Denenen']} (%{row['Yüzde']})")
    zone_str = "\n".join(zone_lines) if zone_lines else "Veri yok"

    system_prompt = (
        "Sen uzman bir NBA taktik analistisin. Kullanıcının sorusunu verilen istatistiklere dayanarak "
        "kısa, net ve akıcı bir TÜRKÇE ile yanıtla. Yanıtını maddeler halinde 3 ana başlıkta topla: "
        "1) Form Durumu 2) Şut & Skor Tehdidi 3) Taktiksel Öneri. Asla döngüye girme."
    )

    user_prompt = f"Oyuncu: {player_name}\n\nSon Maçlar:\n{recent_str}\n\nBölgesel Şut Yüzdeleri:\n{zone_str}\n\nKullanıcı Sorusu: {user_question}"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.85, "num_predict": 350}
    }

    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except Exception as exc:
        return f"AI Bağlantı Hatası: {exc}"


def query_ollama_comparison(p1_name: str, p1_stats: dict, p2_name: str, p2_stats: dict) -> str:
    system_prompt = (
        "Sen profesyonel bir NBA scout ve analistisin. İki oyuncunun sağlanan sezon ve son maç verilerini kıyaslayarak "
        "kısa, net ve akıcı bir TÜRKÇE scouting raporu yaz. "
        "Raporu şu 3 başlıkta düzenle: "
        "1) Skor & Verimlilik Karşılaştırması 2) Şut Haritası & Alan Hakimiyeti 3) Koç Eşleşme Notu (Hangisi Neden Avantajlı?)."
    )

    user_prompt = f"""
OYUNCU 1: {p1_name}
- Ortalamalar: {p1_stats['pts']:.1f} Sayı, {p1_stats['reb']:.1f} Ribaund, {p1_stats['ast']:.1f} Asist, %{p1_stats['fg']:.1f} FG
- Şut Bölgeleri: {p1_stats['zones']}

OYUNCU 2: {p2_name}
- Ortalamalar: {p2_stats['pts']:.1f} Sayı, {p2_stats['reb']:.1f} Ribaund, {p2_stats['ast']:.1f} Asist, %{p2_stats['fg']:.1f} FG
- Şut Bölgeleri: {p2_stats['zones']}

Lütfen iki oyuncuyu taktiksel olarak karşılaştır:"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.85, "num_predict": 400}
    }

    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except Exception as exc:
        return f"AI Bağlantı Hatası: {exc}"


# ============================================================
# STATE YÖNETİMİ
# ============================================================

if "active_player" not in st.session_state:
    st.session_state["active_player"] = "Alperen Sengun"
if "h2h_p1" not in st.session_state:
    st.session_state["h2h_p1"] = "Alperen Sengun"
if "h2h_p2" not in st.session_state:
    st.session_state["h2h_p2"] = "Nikola Jokic"
if "num_games" not in st.session_state:
    st.session_state["num_games"] = 10
if "chart_mode" not in st.session_state:
    st.session_state["chart_mode"] = "Noktasal"
if "last_ai_analysis" not in st.session_state:
    st.session_state["last_ai_analysis"] = ""
if "last_h2h_analysis" not in st.session_state:
    st.session_state["last_h2h_analysis"] = ""

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏀 NBA Local AI Assistant")
st.sidebar.caption(f"Sezon: {CURRENT_SEASON} · Yerel LLM: {OLLAMA_MODEL}")

chart_mode_field = st.sidebar.radio("Şut Haritası Görünümü:", ["Noktasal", "Isı Haritası"], index=0 if st.session_state["chart_mode"] == "Noktasal" else 1)
num_games_field = st.sidebar.slider("İncelenecek Son Maç Sayısı:", min_value=5, max_value=41, value=st.session_state["num_games"], step=1)

if chart_mode_field != st.session_state["chart_mode"] or num_games_field != st.session_state["num_games"]:
    st.session_state["chart_mode"] = chart_mode_field
    st.session_state["num_games"] = num_games_field
    st.rerun()

st.sidebar.divider()
ollama_status = check_ollama_alive()
if ollama_status:
    st.sidebar.success(f"Ollama Aktif ({OLLAMA_MODEL})")
else:
    st.sidebar.error("Ollama Kapalı (localhost:11434)")

# ============================================================
# ANA PANEL - SEKMELER (TABS)
# ============================================================

tab_single, tab_h2h = st.tabs(["👤 Tekli Oyuncu & Akıllı Asistan", "⚔️ Head-to-Head Kıyaslama"])

# ------------------------------------------------------------
# SEKME 1: TEKLİ OYUNCU ANALİZİ & ARAMA
# ------------------------------------------------------------
with tab_single:
    st.title("🏀 Oyuncu Analiz & Taktik Paneli")

    # 1. DOĞRUDAN OYUNCU ARAMA FORMU
    with st.form("single_player_search_form"):
        col_search, col_submit = st.columns([4, 1])
        with col_search:
            player_query_val = st.text_input(
                "🔍 Oyuncu Adı Girin:", 
                value=st.session_state["active_player"],
                placeholder="Örn: Stephen Curry, LeBron James, Alperen Sengun..."
            )
        with col_submit:
            st.write("")
            st.write("")
            search_submitted = st.form_submit_button("Oyuncuyu Getir", width="stretch")

    if search_submitted and player_query_val.strip():
        st.session_state["active_player"] = player_query_val.strip()
        find_player_id.clear()
        fetch_game_log.clear()
        fetch_shot_chart.clear()
        st.rerun()

    # 2. DOĞAL DİL SERBEST SORU GİRİŞİ (ROUTER)
    with st.expander("🤖 veya Doğal Dille Soru Sorarak Oyuncu Bul (AI Router)"):
        col_ai_input, col_ai_btn = st.columns([4, 1])
        with col_ai_input:
            free_query = st.text_input(
                "Sorunuz:",
                placeholder="Örn: Luka Doncic son maçlarda nasıl oynadı, şutları nasıl?",
                label_visibility="collapsed"
            )
        with col_ai_btn:
            ask_button = st.button("Soruyu Yanıtla", type="primary", width="stretch", disabled=not ollama_status)

        if ask_button and free_query.strip():
            with st.spinner("AI Router soruyu inceliyor..."):
                routed = route_user_query(free_query)
                detected_player = routed.get("player_name")
                clean_q = routed.get("clean_question", free_query)

                if detected_player:
                    st.session_state["active_player"] = detected_player
                    find_player_id.clear()
                    fetch_game_log.clear()
                    fetch_shot_chart.clear()
                    st.session_state["trigger_question"] = clean_q
                    st.rerun()
                else:
                    st.session_state["trigger_question"] = free_query

    p_id, verified_name = find_player_id(st.session_state["active_player"])

    if p_id is None:
        st.error(f"'{st.session_state['active_player']}' isimli oyuncu bulunamadı. Lütfen adı kontrol edin.")
    else:
        with st.spinner(f"{verified_name} verileri çekiliyor..."):
            g_log = fetch_game_log(p_id, CURRENT_SEASON)
            s_chart = fetch_shot_chart(p_id, CURRENT_SEASON)

        if g_log.empty:
            st.warning(f"{verified_name} için maç verisi bulunamadı.")
        else:
            recent_g = g_log.tail(st.session_state["num_games"]).copy()

            st.divider()
            st.subheader(f"📊 {verified_name} — Son {len(recent_g)} Maç Ortalamaları")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sayı (PTS)", f"{recent_g['PTS'].mean():.1f}")
            c2.metric("Ribaund (REB)", f"{recent_g['REB'].mean():.1f}")
            c3.metric("Asist (AST)", f"{recent_g['AST'].mean():.1f}")
            c4.metric("Şut İsabeti (FG%)", f"{recent_g['FG_PCT'].mean() * 100:.1f}%")

            st.divider()

            col_shot, col_stats = st.columns([1.15, 1])
            with col_shot:
                st.subheader(f"🎯 Şut Haritası ({st.session_state['chart_mode']})")
                fig = plot_shot_chart(s_chart, title=f"{verified_name} - Şut Dağılımı", mode=st.session_state["chart_mode"])
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            with col_stats:
                st.subheader("📍 Bölgesel Verimlilik")
                z_stats = compute_zone_stats(s_chart)
                if not z_stats.empty:
                    st.dataframe(z_stats, width="stretch", hide_index=True)

                st.markdown("**Son Maçlarda Sayı Trendi**")
                b_data = recent_g[["GAME_DATE", "PTS"]].copy()
                b_data["GAME_DATE"] = b_data["GAME_DATE"].dt.strftime("%d.%m")
                st.bar_chart(b_data.set_index("GAME_DATE"), y="PTS", color="#3ddc84")

            st.divider()

            st.subheader("🧠 Taktiksel AI Analizi")
            
            preset_questions = {
                "🛡️ Rakip Savunma Stratejisi (Scouting Raporu)": (
                    f"Rakip takım koçu olarak {verified_name}'e karşı nasıl bir savunma kurgulamalıyım? "
                    "Şut haritasındaki zayıf ve güçlü noktalarını baz alarak savunma planı çıkar."
                ),
                "🎯 Şut Tercihleri ve Hücum Verimliliği": (
                    f"{verified_name}'in şut seçimleri ne kadar verimli? Boyalı alan, orta mesafe ve 3 sayı tercihlerini değerlendir."
                ),
                "📈 Form Trendi ve Tutarlılık Analizi": (
                    f"{verified_name}'in son maçlardaki skor, ribaund ve asist istikrarı nasıl? Performansı yükselişte mi?"
                ),
                "✍️ Özel Soru Yaz": ""
            }

            selected_preset = st.selectbox("📌 Analiz Konusu Seçin:", list(preset_questions.keys()))
            default_text = preset_questions[selected_preset] if selected_preset != "✍️ Özel Soru Yaz" else f"{verified_name} hakkında sormak istediğiniz taktiksel soru..."

            user_question_input = st.text_area("Taktiksel Soru:", value=default_text, height=80)

            if st.button("🤖 Rapor Üret", type="primary", disabled=not ollama_status):
                with st.spinner("Phi-3 analiz hazırlıyor..."):
                    st.session_state["last_ai_analysis"] = query_ollama_analysis(
                        verified_name, g_log, z_stats, user_question_input
                    )

            if "trigger_question" in st.session_state and st.session_state["trigger_question"]:
                with st.spinner("Phi-3 verileri inceliyor ve analizi yazıyor..."):
                    st.session_state["last_ai_analysis"] = query_ollama_analysis(
                        verified_name, g_log, z_stats, st.session_state["trigger_question"]
                    )
                del st.session_state["trigger_question"]

            if st.session_state["last_ai_analysis"]:
                st.info(st.session_state["last_ai_analysis"])

# ------------------------------------------------------------
# SEKME 2: HEAD-TO-HEAD (KIYASLAMA) MODU
# ------------------------------------------------------------
with tab_h2h:
    st.title("⚔️ NBA Head-to-Head Karşılaştırma")
    st.markdown("İki oyuncunun şut haritalarını, bölge hakimiyetlerini ve maç istatistiklerini yan yana kıyaslayın.")

    col_p1, col_p2, col_h2h_btn = st.columns([2, 2, 1])
    with col_p1:
        p1_input = st.text_input("1. Oyuncu:", value=st.session_state["h2h_p1"])
    with col_p2:
        p2_input = st.text_input("2. Oyuncu:", value=st.session_state["h2h_p2"])
    with col_h2h_btn:
        st.write("")
        st.write("")
        compare_btn = st.button("⚖️ Kıyasla", type="primary", width="stretch")

    if compare_btn:
        st.session_state["h2h_p1"] = p1_input.strip()
        st.session_state["h2h_p2"] = p2_input.strip()
        st.rerun()

    p1_id, p1_name = find_player_id(st.session_state["h2h_p1"])
    p2_id, p2_name = find_player_id(st.session_state["h2h_p2"])

    if not p1_id or not p2_id:
        st.error("Lütfen geçerli iki NBA oyuncu adı girin.")
    else:
        with st.spinner("İki oyuncunun verileri yükleniyor..."):
            p1_log = fetch_game_log(p1_id, CURRENT_SEASON)
            p1_shots = fetch_shot_chart(p1_id, CURRENT_SEASON)
            p2_log = fetch_game_log(p2_id, CURRENT_SEASON)
            p2_shots = fetch_shot_chart(p2_id, CURRENT_SEASON)

        if p1_log.empty or p2_log.empty:
            st.warning("Oyunculardan biri veya her ikisi için yeterli maç kaydı bulunamadı.")
        else:
            p1_recent = p1_log.tail(st.session_state["num_games"])
            p2_recent = p2_log.tail(st.session_state["num_games"])

            st.subheader("📊 İstatistiksel Kıyaslama (Son Maçlar)")
            
            p1_pts, p2_pts = p1_recent['PTS'].mean(), p2_recent['PTS'].mean()
            p1_reb, p2_reb = p1_recent['REB'].mean(), p2_recent['REB'].mean()
            p1_ast, p2_ast = p1_recent['AST'].mean(), p2_recent['AST'].mean()
            p1_fg, p2_fg = p1_recent['FG_PCT'].mean() * 100, p2_recent['FG_PCT'].mean() * 100

            metrics_df = pd.DataFrame({
                "Metrik": ["Sayı (PTS)", "Ribaund (REB)", "Asist (AST)", "Şut İsabeti (FG%)"],
                p1_name: [f"{p1_pts:.1f}", f"{p1_reb:.1f}", f"{p1_ast:.1f}", f"%{p1_fg:.1f}"],
                p2_name: [f"{p2_pts:.1f}", f"{p2_reb:.1f}", f"{p2_ast:.1f}", f"%{p2_fg:.1f}"],
                "Fark / Lider": [
                    f"{abs(p1_pts - p2_pts):.1f} ({p1_name if p1_pts > p2_pts else p2_name})",
                    f"{abs(p1_reb - p2_reb):.1f} ({p1_name if p1_reb > p2_reb else p2_name})",
                    f"{abs(p1_ast - p2_ast):.1f} ({p1_name if p1_ast > p2_ast else p2_name})",
                    f"%{abs(p1_fg - p2_fg):.1f} ({p1_name if p1_fg > p2_fg else p2_name})",
                ]
            })
            st.dataframe(metrics_df, width="stretch", hide_index=True)

            st.divider()

            st.subheader(f"🎯 Şut Haritaları Kıyaslaması ({st.session_state['chart_mode']})")
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                fig1 = plot_shot_chart(p1_shots, title=f"{p1_name} Şut Haritası", mode=st.session_state["chart_mode"])
                st.pyplot(fig1, width="stretch")
                plt.close(fig1)

            with col_chart2:
                fig2 = plot_shot_chart(p2_shots, title=f"{p2_name} Şut Haritası", mode=st.session_state["chart_mode"])
                st.pyplot(fig2, width="stretch")
                plt.close(fig2)

            st.subheader("📍 Bölgesel İsabet Karşılaştırması")
            p1_zones = compute_zone_stats(p1_shots).rename(columns={"Yüzde": f"{p1_name} %", "Denenen": f"{p1_name} Deneme", "İsabetli": f"{p1_name} İsabet"})
            p2_zones = compute_zone_stats(p2_shots).rename(columns={"Yüzde": f"{p2_name} %", "Denenen": f"{p2_name} Deneme", "İsabetli": f"{p2_name} İsabet"})
            
            merged_zones = pd.merge(p1_zones, p2_zones, on="Bölge", how="outer").fillna("-")
            st.dataframe(merged_zones, width="stretch", hide_index=True)

            st.divider()

            st.subheader("🧠 Yerel AI (Phi-3) Head-to-Head Scouting Raporu")
            
            if st.button("🤖 İki Oyuncuyu Kıyasla ve Rapor Üret", type="primary", disabled=not ollama_status):
                with st.spinner("Phi-3 iki oyuncunun verilerini çapraz inceleyip rapor hazırlıyor..."):
                    p1_zone_str = ", ".join([f"{r['Bölge']}: %{r[f'{p1_name} %']}" for _, r in p1_zones.iterrows()]) if not p1_zones.empty else "Veri yok"
                    p2_zone_str = ", ".join([f"{r['Bölge']}: %{r[f'{p2_name} %']}" for _, r in p2_zones.iterrows()]) if not p2_zones.empty else "Veri yok"

                    p1_payload = {'pts': p1_pts, 'reb': p1_reb, 'ast': p1_ast, 'fg': p1_fg, 'zones': p1_zone_str}
                    p2_payload = {'pts': p2_pts, 'reb': p2_reb, 'ast': p2_ast, 'fg': p2_fg, 'zones': p2_zone_str}

                    st.session_state["last_h2h_analysis"] = query_ollama_comparison(p1_name, p1_payload, p2_name, p2_payload)

            if st.session_state["last_h2h_analysis"]:
                st.info(st.session_state["last_h2h_analysis"])