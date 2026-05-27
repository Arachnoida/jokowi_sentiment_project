"""
src/modeling
Tahap modeling: dua jalur sentimen (SVM + TF-IDF, dan IndoBERT fine-tuning).

Catatan desain:
- Data berlabel relatif kecil (target 1.000/kelas = 3.000), jadi seluruh tahap
  ini SENGAJA tidak memakai Spark — cukup pandas/scikit-learn di memori.
- SVM dijalankan lokal; IndoBERT fine-tuning dijalankan di Colab/Kaggle (GPU)
  lewat notebooks/07_indobert_finetune_colab.ipynb.
- Split train/val/test dibuat SEKALI dan dipakai identik oleh kedua model agar
  perbandingan adil.
"""
