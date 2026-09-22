"""Sanitise "class-name-echoing refusals" in TimeOmni-VL understanding outputs (pre-evaluation step).

Problem: TimeOmni-VL emits templated refusals for many classification samples, and the refusal
sentence **lists every candidate class name** (e.g. all 900/900 MarmAudio rows read "The provided
data does not correspond to any of the following call types: Twitter, Tsik, Phee, Trill,
Infant_cry, or Seep."). eval_understanding.classification_eval finds predicted labels by substring
matching (`predicted = {l for l in all_labels if l in gen}`) -- a refusal containing every class name
=> every label counts as "predicted" => the ground truth is always among them => accuracy inflates to
~100% (F1 only reaches ~0.29 because of 5 FPs per row, but any accuracy-based protocol is completely
polluted). Semantically such an answer explicitly says "belongs to none of the classes"; fair scoring
= no prediction (wrong), not "predicted all of them".

Fix: for every row whose generated_text matches a class-name-echoing refusal pattern, replace the
text with the placeholder "[REFUSED]" (contains no class name / keyword => classification matches no
label and scores 0; detection detect() also matches no false/true anchor: sets with a non-empty true
list return None and are dropped, sets with an empty true list (GWOSC/STEAD/TIMECAP) fall back to
True -- refusals on such detection sets are outside the scope of this script; the original text is
kept and scored per protocol, with the refusal rate reported separately).

Only **negations that explicitly enumerate class names** are cleaned (patterns in
_NEG_ECHO_PATTERNS); plain refusals ("I cannot answer...") are left alone -- that is a
detection-protocol issue made transparent through the refusal rate, not by editing data.

Output: same-named jsonl files (cleaned) under --out_dir, for re-scoring with eval_understanding.py;
the number of cleaned rows per file is printed. The original directory is untouched (the raw protocol
stays reproducible).
"""
import argparse
import json
import os
import re

_NEG_ECHO_PATTERNS = [
    re.compile(r"does not correspond to any", re.I),
    re.compile(r"does not (?:match|belong to) any", re.I),
    re.compile(r"none of the (?:following|provided|listed)", re.I),
]


def is_neg_echo(text: str) -> bool:
    return any(p.search(text or "") for p in _NEG_ECHO_PATTERNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for fn in sorted(os.listdir(args.in_dir)):
        if not fn.endswith(".jsonl"):
            continue
        rows = [json.loads(l) for l in open(os.path.join(args.in_dir, fn))]
        n_clean = 0
        for r in rows:
            if is_neg_echo(r.get("generated_text") or ""):
                r["generated_text"] = "[REFUSED]"
                n_clean += 1
        with open(os.path.join(args.out_dir, fn), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{fn}: cleaned {n_clean}/{len(rows)} rows (class-name-echoing refusal -> [REFUSED])")


if __name__ == "__main__":
    main()
