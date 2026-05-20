---
title: Jokowi Label Studio
emoji: 🏷️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Jokowi Sentiment — Label Studio

Anotasi sentimen komentar YouTube terhadap Jokowi (Positif / Negatif / Netral).

- **Engine:** Label Studio (image resmi `heartexlabs/label-studio`).
- **Database:** Neon Postgres (persisten) — di-set via Secrets/Variables di Settings Space.
- **Port:** 7860 (lihat `Dockerfile`).
