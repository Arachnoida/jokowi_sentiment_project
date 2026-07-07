#!/usr/bin/env python3
import os, sys, uno
from com.sun.star.beans import PropertyValue

SRC = os.path.abspath(sys.argv[1]); DST = os.path.abspath(sys.argv[2])
PORT = sys.argv[3] if len(sys.argv) > 3 else "2002"

lc = uno.getComponentContext()
resolver = lc.ServiceManager.createInstanceWithContext(
    "com.sun.star.bridge.UnoUrlResolver", lc)
ctx = resolver.resolve(
    f"uno:socket,host=localhost,port={PORT};urp;StarOffice.ComponentContext")
smgr = ctx.ServiceManager
desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)

def pv(n, v):
    p = PropertyValue(); p.Name = n; p.Value = v; return p

doc = desktop.loadComponentFromURL(
    uno.systemPathToFileUrl(SRC), "_blank", 0, (pv("Hidden", True),))
try:
    idxs = doc.getDocumentIndexes()
    for i in range(idxs.getCount()):
        idxs.getByIndex(i).update()
    print("index updated:", idxs.getCount(), flush=True)
except Exception as e:
    print("warn index:", e, flush=True)
try:
    doc.refresh()
except Exception as e:
    print("warn refresh:", e, flush=True)

doc.storeToURL(uno.systemPathToFileUrl(DST),
               (pv("FilterName", "writer_pdf_Export"),))
doc.close(False)
print("PDF OK:", DST, flush=True)
