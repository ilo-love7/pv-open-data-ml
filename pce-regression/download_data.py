# -*- coding: utf-8 -*-
"""下载 Perovskite Database 全量 CSV（84.7MB）到 data/ 目录。

来源：GitHub Jesperkemist/perovskitedatabase_data（Zenodo DOI 10.5281/zenodo.5834698）
CC-BY 4.0 开源。若网络受限（直连超时），见下方"备用方案"。

运行：python download_data.py
"""

# %%
import os
import urllib.request

URL = ("https://media.githubusercontent.com/media/Jesperkemist/"
       "perovskitedatabase_data/main/data/Perovskite_database_content_all_data.csv")
DEST = os.path.join("data", "perovskite_database_all.csv")

os.makedirs("data", exist_ok=True)

# 断点续传
start = os.path.getsize(DEST) if os.path.exists(DEST) else 0
req = urllib.request.Request(URL)
if start:
    req.add_header("Range", f"bytes={start}-")
    print(f"续传，从 {start/1e6:.1f} MB 处继续…")

with urllib.request.urlopen(req, timeout=60) as r, \
        open(DEST, "ab" if start else "wb") as f:
    total = int(r.headers.get("Content-Length", 0)) + start
    done = start
    while True:
        chunk = r.read(1 << 20)
        if not chunk:
            break
        f.write(chunk)
        done += len(chunk)
        print(f"\r{done/1e6:7.1f} / {total/1e6:.1f} MB", end="", flush=True)
print("\n完成 ->", DEST)

# 备用方案：直连被墙时，用 12 路分段并行（在 Git Bash 里执行，约 4 分钟）：
#   SIZE=84659951; N=12; CHUNK=$(( (SIZE+N-1)/N )); URL="<上面的地址>"
#   for i in $(seq 0 11); do
#     S=$((i*CHUNK)); E=$(( (i+1)*CHUNK-1 )); [ $E -ge $SIZE ] && E=$((SIZE-1))
#     curl -sS -L --retry 5 -r $S-$E -o part_$i "$URL" &
#   done; wait; cat part_00 ... part_11 > data/perovskite_database_all.csv
