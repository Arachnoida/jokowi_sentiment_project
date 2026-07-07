#!/usr/bin/env python3
"""Tambah footer nomor halaman + section body (angka mulai 1). Cover & daftar-isi
section sudah diatur di document.xml (dari markdown). Di sini: footer part, rel,
content-type, dan modif sectPr terakhir (body) -> footer + pgNumType decimal."""
import sys, re, zipfile, shutil
docx=sys.argv[1]
FID="rId990"
with zipfile.ZipFile(docx) as z:
    data={n:z.read(n) for n in z.namelist()}

# 1) footer part (PAGE field, center, TNR 12)
FOOTER=('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
 '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
 '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
 '<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>'
 '<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
 '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
 '<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr><w:t>1</w:t></w:r>'
 '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')
data['word/footer1.xml']=FOOTER.encode('utf-8')

# 2) rel
rels=data['word/_rels/document.xml.rels'].decode('utf-8')
if FID not in rels:
    rels=rels.replace('</Relationships>',
      f'<Relationship Id="{FID}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/></Relationships>')
data['word/_rels/document.xml.rels']=rels.encode('utf-8')

# 3) content-type override
ct=data['[Content_Types].xml'].decode('utf-8')
if 'footer1.xml' not in ct:
    ct=ct.replace('</Types>',
      '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>')
data['[Content_Types].xml']=ct.encode('utf-8')

# 4) sectPr body terakhir: footerReference + pgNumType decimal start 1
xml=data['word/document.xml'].decode('utf-8')
i=xml.rfind('<w:sectPr')
j=xml.index('>', i)
open_tag=xml[i:j+1]
rest=xml[j+1:]
se=rest.index('</w:sectPr>')
body=rest[:se]; after=rest[se:]
if 'footerReference' not in body:
    open_tag=open_tag+f'<w:footerReference w:type="default" r:id="{FID}"/>'
m=re.search(r'<w:pgMar[^/]*/>', body)
pgnum='<w:pgNumType w:fmt="decimal" w:start="1"/>'
if 'pgNumType' not in body:
    body = (body[:m.end()]+pgnum+body[m.end():]) if m else (body+pgnum)
xml=xml[:i]+open_tag+body+after
data['word/document.xml']=xml.encode('utf-8')

tmp=docx+".new"
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
    for n,b in data.items(): z.writestr(n,b)
shutil.move(tmp,docx)
print("footer + section body (nomor angka) OK")
