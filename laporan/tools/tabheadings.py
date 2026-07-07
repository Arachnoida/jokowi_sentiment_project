#!/usr/bin/env python3
"""Sisipkan <w:tab/> antara nomor subbab (N.N) dan judulnya di paragraf Heading2/3."""
import sys, re, zipfile, os, shutil

docx = sys.argv[1]
with zipfile.ZipFile(docx) as z:
    data = {n: z.read(n) for n in z.namelist()}
xml = data['word/document.xml'].decode('utf-8')

def fix_para(m):
    para = m.group(0)
    if 'w:val="Heading2"' not in para and 'w:val="Heading3"' not in para:
        return para
    done = {"n": 0}
    def rep_t(t):
        if done["n"]: return t.group(0)
        inner = t.group(2)
        mm = re.match(r'^(\d+(?:\.\d+)+)\s+(.+)$', inner, re.S)
        if not mm: return t.group(0)
        done["n"] = 1
        return (f'<w:t xml:space="preserve">{mm.group(1)}</w:t>'
                f'<w:tab/><w:t xml:space="preserve">{mm.group(2)}</w:t>')
    return re.sub(r'(<w:t[^>]*>)([^<]*)(</w:t>)', rep_t, para)

new = re.sub(r'<w:p\b.*?</w:p>', fix_para, xml, flags=re.S)
data['word/document.xml'] = new.encode('utf-8')

tmp = docx + ".new"
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
    for n, b in data.items():
        z.writestr(n, b)
shutil.move(tmp, docx)
print("tab disisipkan di heading subbab")
