# YPO Analytics Streamlit Dashboard V5

Dashboard analisis sentimen YouTube berbasis Streamlit dan Plotly untuk eksplorasi dataset, evaluasi model, error analysis, public opinion insight, dan metodologi penelitian.

## Menjalankan Aplikasi

```bash
python -m streamlit run streamlit_app/app.py
```

Dashboard membaca data dari folder:

```text
outputs/tableau/
```

Folder tersebut perlu berisi file CSV data mart hasil pipeline export.

## Fokus V5

V5.2 memoles dashboard pada aspek visual hierarchy, spacing, metric cards, sidebar, Plotly theme, data table, methodology cards, dan konsistensi design system. Patch ini juga mempertahankan perbaikan Reset Filters dari V4.1 dan menambahkan fallback aman pada halaman Error Analysis ketika data prediksi detail model tertentu tidak tersedia.


V5.2: memperbaiki native sidebar reopen control pada Streamlit dengan menjaga layer header tetap hidup namun transparan dan non-blocking.
