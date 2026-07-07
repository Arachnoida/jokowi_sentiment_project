#!/usr/bin/env python3
"""Regenerate semua figur (2 model, TANPA em dash): arsitektur, bar chart, 2 confusion matrix."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np, json, os

ROOT="/home/ravi/Projects/jokowi_sentiment_project"
OUT=os.path.join(ROOT,"outputs"); REP=os.path.join(OUT,"reports")

# ---------- ARSITEKTUR ----------
fig,ax=plt.subplots(figsize=(8.6,10.4)); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
def box(x,y,w,h,title,body,fc,ec):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.6,rounding_size=2.2",fc=fc,ec=ec,lw=1.6))
    cx=x+w/2
    ax.text(cx,y+h-2.6,title,ha="center",va="top",fontsize=9.5,weight="bold",color=ec)
    ax.text(cx,y+h-6.0,body,ha="center",va="top",fontsize=7.4,color="#333333")
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=13,lw=1.4,color="#5A6B7B"))
BLUE=("#DAE8FC","#3B6FB0"); PURP=("#E1D5E7","#8259A0"); GREEN=("#D5E8D4","#5B9E52")
ORNG=("#FFE6CC","#D79B00"); YELL=("#FFF2CC","#C9A227"); GREY=("#F0F0F0","#8C8C8C")
ax.text(50,98,"Pipeline Analisis Sentimen Komentar YouTube - Isu Ijazah Jokowi",ha="center",va="top",fontsize=11.5,weight="bold",color="#1a1a1a")
ax.text(50,94.2,"SVM + TF-IDF  vs  IndoBERTweet   ·   3 kelas: Positif / Negatif / Netral",ha="center",va="top",fontsize=8.6,color="#7A7A7A")
box(18,84,64,8.0,"1 · PENGUMPULAN DATA","YouTube Data API  ->  koleksi raw_comments (14.107 komentar)\nnotebook: ingestion.ipynb",*BLUE)
box(15,69.5,70,10.5,"2 · PELABELAN","Pelabelan LLM (\"claude-llm\") · 10.000 komentar\nrubrik issue-anchored: sikap thd narasi tuduhan (bukan sosok Jokowi)\ndataset final: seimbang 3.000 (1.000/kelas), label diaudit (v1audited)",*PURP)
box(6,54,42,11.0,"3a · PREPROCESSING - SVM","clean agresif + normalisasi slang\n+ buang stopword (negasi dipertahankan)\n+ stemming (Sastrawi)\n-> koleksi processed_svm",*GREEN)
box(52,54,42,11.0,"3b · PREPROCESSING - IndoBERTweet","cleaning minimal\n(morfologi · tanda baca · negasi utuh)\n\n-> koleksi processed_bert",*GREEN)
box(6,40,42,10.0,"4a · MODELING - SVM + TF-IDF","TF-IDF -> LinearSVC\ntuning di validation · LOKAL / CPU",*ORNG)
box(52,40,42,10.0,"4b · MODELING - IndoBERTweet","fine-tune indobertweet-base-uncased\nCOLAB / GPU",*ORNG)
box(15,23,70,11.0,"5 · HASIL & PERBANDINGAN","SVM + TF-IDF  vs  IndoBERTweet  (metrik utama: macro-F1)\ntest set identik (300 komentar, seimbang)\nIndoBERTweet 0,8721  >  SVM + TF-IDF 0,8556  (macro-F1)",*YELL)
ax.add_patch(FancyBboxPatch((5,11),90,5.6,boxstyle="round,pad=0.5,rounding_size=1.5",fc=GREY[0],ec=GREY[1],lw=1.2))
ax.text(50,13.8,"Penyimpanan terpusat: MongoDB Atlas - DB \"youtube_sentiment\"  (raw_comments · processed_svm · processed_bert) · semua data = dokumen JSON",ha="center",va="center",fontsize=7.2,color="#555555")
arrow(50,84,50,80.2); arrow(40,69.4,27,65.2); arrow(60,69.4,73,65.2)
arrow(27,54,27,50.2); arrow(73,54,73,50.2); arrow(27,40,42,34.2); arrow(73,40,58,34.2)
ax.text(50,37.4,"Split kanonik identik per model (urut comment_id + seed=42)  ->  perbandingan adil",ha="center",va="center",fontsize=7.0,style="italic",color="#C0791A")
plt.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01)
plt.savefig(os.path.join(OUT,"architecture_2model.png"),dpi=150,bbox_inches="tight",facecolor="white"); plt.close()
print("arch OK")

# ---------- BAR CHART ----------
svm=json.load(open(os.path.join(REP,"svm_v1audited_metrics.json")))["test"]
ibt=json.load(open(os.path.join(REP,"indobertweet_v1audited_metrics.json")))["test"]
models=["SVM + TF-IDF\n(sklearn)","IndoBERTweet"]
acc=[svm["accuracy"],ibt["accuracy"]]; mf1=[svm["macro_f1"],ibt["macro_f1"]]
fig,ax=plt.subplots(figsize=(7.2,4.6)); x=np.arange(2); w=0.36
b1=ax.bar(x-w/2,acc,w,label="Akurasi",color="#4C78A8",edgecolor="#2f4b6e")
b2=ax.bar(x+w/2,mf1,w,label="Macro-F1",color="#E1812C",edgecolor="#9c5716")
for bars in (b1,b2):
    for r in bars: ax.text(r.get_x()+r.get_width()/2,r.get_height()+0.008,f"{r.get_height():.4f}",ha="center",va="bottom",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(models,fontsize=10); ax.set_ylim(0,1.12)
ax.set_ylabel("Skor (test set, n=300)")
ax.set_title("Perbandingan Model (v1audited): Akurasi & Macro-F1",fontsize=11,weight="bold",pad=26)
ax.legend(loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(0.5,1.02))
ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",ls=":",alpha=0.4)
plt.tight_layout(); plt.savefig(os.path.join(REP,"model_comparison_2model.png"),dpi=150,facecolor="white"); plt.close()
print("bar OK")

# ---------- CONFUSION MATRICES (bersih, tanpa em dash) ----------
def cm_plot(metrics_file, title, out, cmap):
    m=json.load(open(os.path.join(REP,metrics_file)))["test"]
    cm=np.array(m["confusion_matrix"]); labels=["Negatif","Netral","Positif"]
    acc=m["accuracy"]
    fig,ax=plt.subplots(figsize=(5.2,4.6))
    im=ax.imshow(cm,cmap=cmap,vmin=0,vmax=cm.max())
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    thr=cm.max()*0.6
    for i in range(3):
        for j in range(3):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=13,
                    color="white" if cm[i,j]>thr else "#222222")
    ax.set_title(f"{title}  (test, akurasi {acc:.3f})",fontsize=11,weight="bold",pad=10)
    plt.tight_layout(); plt.savefig(os.path.join(REP,out),dpi=150,facecolor="white"); plt.close()
    print(out,"OK")
cm_plot("svm_v1audited_metrics.json","Confusion Matrix SVM + TF-IDF","svm_cm_clean.png","Blues")
cm_plot("indobertweet_v1audited_metrics.json","Confusion Matrix IndoBERTweet","indobertweet_cm_clean.png","Purples")
