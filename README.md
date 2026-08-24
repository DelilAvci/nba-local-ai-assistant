# 🏀 NBA Game & Player Assistant Agent (Local AI)

Tamamen yerel donanımda çalışan, bulut bağımlılığı olmayan interaktif NBA Analitik ve Scouting Asistanı. `nba_api` üzerinden gerçek zamanlı maç verilerini çeker, resmi saha koordinatlarında şut analizi yapar ve **Microsoft Phi-3 (Ollama)** modeliyle taktiksel Türkçe raporlar üretir.

## 🚀 Özellikler
- **Doğal Dil Router (Tool-Calling):** Kullanıcı sorusundan oyuncu adını ve amacını otomatik tespit eder.
- **Resmi NBA Şut Haritası:** Matplotlib ile interaktif Noktasal (Scatter) ve Yoğunluk (Hexbin Isı Haritası) görselleştirmesi.
- **Bölgesel Verimlilik:** Boyalı alan, orta mesafe ve 3 sayı bölgelerinin isabet yüzdeleri.
- **Head-to-Head (Kıyaslama Modu):** İki oyuncunun şut haritalarını, metrik liderliklerini ve taktiksel scouting raporlarını yan yana karşılaştırma.
- **Edge / Local AI:** Ollama üzerinden çalışan `phi3:mini` ile düşük gecikmeli, yerel çıkarım.

## 🛠️ Kurulum

1. **Repoyu klonlayın ve dizine geçin:**
   ```bash
   git clone <REPO_URL>
   cd nba_local_agent