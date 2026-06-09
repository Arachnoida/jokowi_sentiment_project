"""Jalur PySpark untuk proyek sentimen Jokowi.

Paket ini memigrasikan pipeline preprocessing & pelatihan SVM dari notebook
(.ipynb) ke skrip .py berbasis Spark, sebagai demonstrasi tooling Big Data.

Catatan metodologi (penting untuk skripsi):
  - Dataset hanya 14.107 baris -> di-run dengan Spark mode lokal ``local[*]``.
    Spark di sini untuk MEMPERLIHATKAN pipeline distributed (Tokenizer ->
    CountVectorizer -> IDF -> OneVsRest(LinearSVC)), bukan karena volume data
    menuntutnya.
  - Split train/val/test dihitung SEKALI secara deterministik (identik dengan
    jalur sklearn ``train_svm_full14k.py`` dan jalur IndoBERT) lalu di-join ke
    Spark DataFrame, supaya test set tetap apple-to-apple lintas model.
  - Fine-tune IndoBERT TETAP di HuggingFace/PyTorch (Colab GPU); Spark MLlib
    tidak menyediakan jalur fine-tune transformer. Batas ini wajar & dicatat.
"""
