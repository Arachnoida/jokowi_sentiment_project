#!/usr/bin/env python3
"""Buka docx, refresh TOC, simpan ulang sebagai docx Word-friendly (paket standar)."""
import os, sys, uno
from com.sun.star.beans import PropertyValue
SRC=os.path.abspath(sys.argv[1]); DST=os.path.abspath(sys.argv[2]); PORT=sys.argv[3]
lc=uno.getComponentContext()
res=lc.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", lc)
ctx=res.resolve(f"uno:socket,host=localhost,port={PORT};urp;StarOffice.ComponentContext")
desktop=ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
def pv(n,v):
    p=PropertyValue(); p.Name=n; p.Value=v; return p
doc=desktop.loadComponentFromURL(uno.systemPathToFileUrl(SRC),"_blank",0,(pv("Hidden",True),))
try:
    idxs=doc.getDocumentIndexes()
    for i in range(idxs.getCount()): idxs.getByIndex(i).update()
    doc.refresh()
except Exception as e: print("warn:",e,flush=True)
# simpan sebagai docx (filter Word)
try:
    doc.storeToURL(uno.systemPathToFileUrl(DST),(pv("FilterName","MS Word 2007 XML"),))
    print("saved MS Word 2007 XML",flush=True)
except Exception:
    doc.storeToURL(uno.systemPathToFileUrl(DST),(pv("FilterName","Office Open XML Text"),))
    print("saved Office Open XML Text",flush=True)
doc.close(False)
print("DOCX OK:",DST,flush=True)
