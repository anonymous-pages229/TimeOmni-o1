"""Build the 12-domain single-pool training data -- no focus/rest distinction.

Composition (`domain_of` classifies each row; same convention as the main table):
- the 9 small/medium domains: **all rows** (ST-Bench rows have think cleared = standing convention);
- **small domains are upsampled to >= 1000 rows**: Bioacoustics 127->1000, Energy 198->1000,
  Meteorology 993->1000 (topped up by repeated sampling with seed 17; shuffled with the other rows
  at training time);
- the three largest domains are **downsampled** (reservoir sampling with seed 17, natural source
  proportions within a domain; the differentiated caps follow the measured forgetting sensitivity):
  Physiology 241,804->20,000 (a 1947-row anchor was far from enough),
  Telecommunications 154,185->8,000 (only -2.1 at 1947 rows),
  Cross_domain 49,324->8,000 (-0.05 at 1947 rows);
- the 82 "Unlabeled" rows are dropped.
About 90,247 rows per epoch in total.

    python chronos_llm/scripts/utils/make_domain12_pool.py   # streaming, lightweight
"""
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from chronos_llm.scripts.utils.summarize_agg6_by_domain import domain_of  # noqa: E402

BASE = "data/understanding"
SOURCES = [
    f"{BASE}/veritime/veritime_train.jsonl",
    f"{BASE}/time-ra/time_ra_train.jsonl",
    f"{BASE}/opentslm/ecg_qa_train.jsonl",
    f"{BASE}/opentslm/har_train.jsonl",
    f"{BASE}/opentslm/sleep_train.jsonl",
    f"{BASE}/telecomts/telecomts_train.jsonl",
    f"{BASE}/hitsr/hitsr_train.jsonl",
    f"{BASE}/st-bench/stbench_train.jsonl",
]
CAP = {"Physiology": 20000, "Telecommunications": 8000, "Cross_domain_Synthetic": 8000}
UPSAMPLE_TO = 1000        # domains with fewer rows than this are topped up by repeated sampling
OUT_DIR = f"{BASE}/derived/dom12"
os.makedirs(OUT_DIR, exist_ok=True)

rng = random.Random(17)
reservoir = {d: [] for d in CAP}
seen = {d: 0 for d in CAP}
small = {}                # domain -> list of rows (small/medium domains are written first; rows kept in memory only for possible upsampling)
n_keep = n_drop = n_nothink = 0
keep_counts = {}

out_path = f"{OUT_DIR}/train_dom12.jsonl"
with open(out_path, "w") as fo:
    for src in SOURCES:
        is_stb = "stbench" in os.path.basename(src)
        for line in open(src):
            row = json.loads(line)
            it = row.get("input_text")
            probe = dict(row, input_text="\n".join(map(str, it)) if isinstance(it, list) else it)
            d = domain_of(probe)
            if d == "Unlabeled":
                n_drop += 1
                continue
            if d in CAP:
                seen[d] += 1
                pool = reservoir[d]
                if len(pool) < CAP[d]:
                    pool.append(line)
                else:
                    j = rng.randint(0, seen[d] - 1)
                    if j < CAP[d]:
                        pool[j] = line
                continue
            if is_stb and row.get("think"):
                row["think"] = ""             # strip reasoning from ST-Bench (standing convention)
                line = json.dumps(row, ensure_ascii=False) + "\n"
                n_nothink += 1
            fo.write(line if line.endswith("\n") else line + "\n")
            n_keep += 1
            keep_counts[d] = keep_counts.get(d, 0) + 1
            small.setdefault(d, []).append(line) if keep_counts[d] <= UPSAMPLE_TO else None
    # Small-domain upsampling: domains below UPSAMPLE_TO are topped up by repeated sampling
    for d, cnt in sorted(keep_counts.items()):
        if cnt < UPSAMPLE_TO:
            extra = [small[d][rng.randrange(cnt)] for _ in range(UPSAMPLE_TO - cnt)]
            for line in extra:
                fo.write(line if line.endswith("\n") else line + "\n")
            n_keep += len(extra)
            print(f"upsampled {d}: {cnt} -> {UPSAMPLE_TO}")
    # Write out the reservoirs of the three large domains
    for d in sorted(CAP):
        for line in reservoir[d]:
            fo.write(line if line.endswith("\n") else line + "\n")
        n_keep += len(reservoir[d])
        print(f"downsampled {d}: {seen[d]} -> {len(reservoir[d])}")

print(f"total {n_keep} rows (ST-Bench think stripped {n_nothink}, unlabeled dropped {n_drop}) -> {out_path}")
