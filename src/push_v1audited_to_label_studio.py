"""Buat project Label Studio khusus dataset v1audited + impor 3.000 komentar
beserta labelnya sebagai ANOTASI (untuk review manual).

Berbeda dgn push_labels_to_label_studio (yg push ke project existing via comment_id),
skrip ini MEMBUAT project baru berisi hanya baris v1audited, dengan label terisi
sebagai anotasi — bisa difilter per kelas, dibrowse, & diedit di UI.

Sumber: outputs/labeling/balanced_3000_v1audited.csv (comment_id, label, confidence, text).

  python -m src.push_v1audited_to_label_studio --dry-run
  python -m src.push_v1audited_to_label_studio --commit
  python -m src.push_v1audited_to_label_studio --commit --source outputs/labeling/balanced_3000_v1sonnet.csv --title "v1 pristine (review)"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from configs.config import Config
from src.push_labels_to_label_studio import LSClient, _read_token, _result_payload
from src.utils import setup_logger

logger = setup_logger("push_v1audited_ls")
ROOT = Path(__file__).resolve().parent.parent
VALID = {"Positif", "Negatif", "Netral"}

LABEL_CONFIG = """<View>
  <Header value="Sikap komentar terhadap NARASI tuduhan 'ijazah Jokowi palsu' (BUKAN sosok Jokowi)"/>
  <View style="color:#6b7280;font-size:13px;margin-bottom:.25em">
    <Text name="r1" value="🟢 Positif = percaya/dukung tuduhan (ijazah palsu)"/>
    <Text name="r2" value="🔴 Negatif = tolak/bantah tuduhan (asli / fitnah)"/>
    <Text name="r3" value="⚪ Netral = sikap tak jelas / bertanya / info saja"/>
    <Text name="conf" value="LLM confidence: $confidence"/>
  </View>
  <View style="padding:1em;background:#f7f8fa;border-radius:8px;font-size:17px;line-height:1.6">
    <Text name="text" value="$text"/>
  </View>
  <Choices name="sentiment" toName="text" choice="single-radio" required="true" showInLine="true">
    <Choice value="Positif" hotkey="1" background="#16a34a"/>
    <Choice value="Negatif" hotkey="2" background="#dc2626"/>
    <Choice value="Netral"  hotkey="3" background="#6b7280"/>
  </Choices>
</View>"""


def _read_rows(path: Path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            lab = (r.get("label") or "").strip()
            cid = (r.get("comment_id") or "").strip()
            if lab in VALID and cid:
                rows.append({
                    "comment_id": cid,
                    "text": r.get("text") or "",
                    "confidence": r.get("confidence") or "",
                    "label": lab,
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="outputs/labeling/balanced_3000_v1audited.csv")
    ap.add_argument("--title", default="v1audited — review (LLM + audit)")
    ap.add_argument("--commit", action="store_true", help="Eksekusi (default dry-run).")
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()

    rows = _read_rows(ROOT / args.source)
    from collections import Counter
    dist = Counter(r["label"] for r in rows)
    print(f"Sumber: {args.source} | {len(rows)} baris | dist: {dict(dist)}")

    if not args.commit:
        print(f"[dry-run] akan BUAT project '{args.title}' + impor {len(rows)} task ber-anotasi.")
        print("Tambah --commit untuk eksekusi.")
        return

    c = LSClient(Config.label_studio.URL, _read_token(None))
    r = c.request("POST", "/api/projects/", json={"title": args.title, "label_config": LABEL_CONFIG})
    if r.status_code not in (200, 201):
        raise SystemExit(f"Gagal buat project: {r.status_code} {r.text[:200]}")
    pid = r.json()["id"]
    print(f"Project dibuat: id={pid}")

    tasks = [{
        "data": {"comment_id": x["comment_id"], "text": x["text"], "confidence": x["confidence"]},
        "annotations": [{"result": _result_payload(x["label"])}],
    } for x in rows]

    total = 0
    for i in range(0, len(tasks), args.chunk):
        batch = tasks[i:i + args.chunk]
        r = c.request("POST", f"/api/projects/{pid}/import", json=batch)
        if r.status_code not in (200, 201):
            raise SystemExit(f"Gagal impor batch {i}: {r.status_code} {r.text[:200]}")
        total += r.json().get("task_count", len(batch))
        print(f"  impor {i}..{i+len(batch)} -> ok (kumulatif ~{total})")

    print(f"\nSELESAI. {total} task ber-anotasi di project id={pid}.")
    print(f"Buka: {Config.label_studio.URL}/projects/{pid}/data")


if __name__ == "__main__":
    main()
