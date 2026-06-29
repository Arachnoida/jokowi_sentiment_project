# IndoBERT — dataset BALANCED 3000 (Colab GPU)

Eksperimen kedua: latih IndoBERT pada subset **balanced 1000/kelas** (3000 total),
dibangun dari label LLM **confidence tertinggi** (`outputs/labeling/balanced_3000.csv`).
Split kanonik identik SVM (urut comment_id, seed=42, 70/20/10 → 2100/600/300).

## Cara A — repo script (paling mudah)

Di Colab, setelah `git pull` repo terbaru (sudah berisi `balanced_3000.csv` +
`train_indobert.py` yang mendukung `--subset`/`--tag`) dan `MONGO_URI` ter-set:

```bash
!python -m src.modeling.train_indobert \
    --subset outputs/labeling/balanced_3000.csv \
    --tag balanced3k
```

Output: `outputs/reports/indobert_balanced3k_metrics.json` +
`indobert_balanced3k_test_confusion.png`. **Commit/kirim JSON itu** → langkah
perbandingan 3 model otomatis.

## Cara B — notebook (jika tidak pakai repo script)

`processed_bert` di Mongo TIDAK punya kolom confidence, jadi subset harus difilter
pakai allowlist `comment_id`. Setelah memuat `df` dari `processed_bert`, sisipkan
SEBELUM split:

```python
import pandas as pd
ids = set(pd.read_csv("balanced_3000.csv", usecols=["comment_id"])["comment_id"].astype(str))
df = df[df["comment_id"].astype(str).isin(ids)].reset_index(drop=True)
# lanjut: df.sort_values("comment_id") -> train_test_split 0.10 -> 0.20/0.90, seed=42
```

(Upload `balanced_3000.csv` ke sesi Colab, atau baca dari repo.)
Simpan hasil test sebagai `indobert_balanced3k_metrics.json` dgn struktur:
`{"model":"IndoBERT","dataset":"balanced3k","test":{accuracy,macro_f1,per_class,confusion_matrix,labels}}`.

## Setelah JSON ada

```bash
python -m src.modeling.compare_models --tag balanced3k
```
→ `model_comparison_balanced3k.csv` + `model_comparison_balanced3k_accuracy.png`.
