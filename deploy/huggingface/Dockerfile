# Label Studio untuk Hugging Face Spaces (SDK: docker), DB di Neon Postgres (persisten).
#
# Catatan penting:
# - Filesystem HF Spaces bersifat SEMENTARA (reset tiap rebuild). Karena itu seluruh
#   anotasi disimpan di Neon Postgres lewat env var POSTGRE_* (di-set sebagai Secrets/
#   Variables di Settings Space), bukan di disk lokal. Folder lokal hanya untuk file
#   sementara.
# - Kredensial Neon TIDAK ditaruh di sini; di-inject saat runtime dari Settings Space.

FROM heartexlabs/label-studio:latest

USER root

# Folder data lokal yang pasti bisa ditulis (container HF bisa jalan dgn UID tak tentu).
RUN mkdir -p /tmp/ls-data && chmod -R 777 /tmp/ls-data

ENV LABEL_STUDIO_BASE_DATA_DIR=/tmp/ls-data \
    DJANGO_DB=default \
    PGSSLMODE=require

# HF Spaces meneruskan trafik publik ke port ini (samakan dgn app_port di README.md).
EXPOSE 7860

# Lewati entrypoint bawaan (yg meng-chown /label-studio); jalankan langsung.
ENTRYPOINT []
CMD ["label-studio", "start", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
