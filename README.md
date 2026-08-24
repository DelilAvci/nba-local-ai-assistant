# 🏀 NBA Game & Player Assistant Agent (Local AI - v3)

Tamamen yerel donanımda çalışan, bulut bağımlılığı olmayan interaktif NBA Analitik, Scouting ve Taktik Asistanı. `nba_api` üzerinden gerçek zamanlı maç ve şut koordinatlarını çeker, resmi saha haritaları üzerinde görselleştirir ve yerel **Qwen 2.5 (qwen2.5:7b)** modeliyle Türkçe taktiksel scouting analizleri üretir.
[![Watch Video](https://img.shields.io/badge/Demo_Video-Google_Drive-blue?style=for-the-badge&logo=google-drive)](https://drive.google.com/file/d/1-QuoSDTm-1RtGcdnK-rt8iRQWZywMt_z/view?usp=sharing)
---

## 🚀 Öne Çıkan Özellikler

- **🤖 Akıllı NLP Router (Intent Parsing):** Doğal dilde yazılan sorulardan oyuncu adını ve analiz amacını sıfır gecikmeyle JSON formatında ayıklar.
- **🗓️ Dinamik Sezon Desteği:** 2021-22'den 2024-25'e kadar geçmiş ve güncel sezon verileri arasında anında geçiş.
- **🎯 Resmi NBA Şut Haritası:** Matplotlib ile interaktif **Noktasal (Scatter)** ve **Yoğunluk (Hexbin Isı Haritası)** görselleştirmeleri.
- **📍 Bölgesel İsabet Matrisi:** Boyalı alan, orta mesafe ve 3 sayı çizgisi arkasındaki şut başarı yüzdeleri.
- **⚔️ Head-to-Head (Kıyaslama Modu):** İki oyuncunun şut dağılımlarını, metrik liderliklerini ve çapraz scouting raporunu yan yana karşılaştırma.
- **🛡️ Dayanıklı API Mimarisi:** Ağ gecikmeleri ve zaman aşımlarına karşı 2x otomatik deneme (retry) ve dinamik `st.toast` bildirimleri.
- **🔒 Tamamen Yerel (Local / Edge AI):** Ollama üzerinden çalışan `qwen2.5:7b` ile veri gizliliği ve sıfır API maliyeti.

---

## 🛠️ Kurulum & Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone [https://github.com/KULLANICI_ADINIZ/nba-local-ai-assistant.git](https://github.com/DelilAvci/nba-local-ai-assistant.git)
cd nba-local-ai-assistant