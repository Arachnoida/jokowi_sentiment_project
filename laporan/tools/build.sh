#!/bin/bash
# Baseplate build: markdown -> docx (reference-doc) -> tab heading -> footer/section
# -> UNO refresh TOC -> PDF + docx Word-friendly.
# Pakai: bash laporan/tools/build.sh [path/ke/laporan.md] [port]
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # laporan/tools
LAP="$(dirname "$HERE")"                                # laporan
ROOT="$(dirname "$LAP")"                                # akar project
MD="${1:-$LAP/BD_UAS_laporan.md}"
NAME="$(basename "$MD" .md)"
REF="$LAP/.template_reference.docx"
PORT="${2:-2960}"
WORK="$(mktemp -d)"

cd "$ROOT"
pandoc "$MD" -o "$WORK/out.docx" --reference-doc="$REF" --resource-path=".:$LAP"
python3 "$HERE/tabheadings.py" "$WORK/out.docx"     # nomor subbab + tab ke judul
python3 "$HERE/pagenumbers.py" "$WORK/out.docx"     # footer + section (cover/romawi/angka)

setsid soffice --headless --invisible --nologo --norestore --nodefault \
  -env:UserInstallation="file://$WORK/lo" \
  "--accept=socket,host=localhost,port=$PORT;urp;" >"$WORK/so.log" 2>&1 </dev/null &
for i in $(seq 1 40); do
  python3 -c "import socket;socket.create_connection(('localhost',$PORT),0.3).close()" 2>/dev/null && break
  sleep 0.5
done
python3 -u "$HERE/uno_convert.py"  "$WORK/out.docx" "$LAP/$NAME.pdf"  "$PORT"   # PDF (TOC refreshed)
python3 -u "$HERE/uno_savedocx.py" "$WORK/out.docx" "$LAP/$NAME.docx" "$PORT"   # docx Word-friendly
echo "OK -> $LAP/$NAME.pdf"
echo "OK -> $LAP/$NAME.docx"
