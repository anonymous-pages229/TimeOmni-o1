"""Mismatched-pairing null-hypothesis control for the TimeOmni-o1 "text generation quality" table.

Background: the "text generation quality" table in the paper only reports TimeOmni-o1's own score
(gen_text paired with the gt_conclusion/gt_roi of its own real sample); it lacks a null-hypothesis
reference answering "is this high score a coincidence, is the judge strict enough?". This script
re-pairs every sample's ``gen_text`` with the ``gt_conclusion`` **and** ``gt_roi`` of **another
sample** (default: a uniformly random derangement over the whole pool, guaranteeing no self-pairing;
``--pairing shift`` exists only to reproduce historical numbers). The two are swapped together as one
"reference truth package" -- swapping only the text but not the ROI values would leave the ROI
overlap-rate / IoU columns identical to the real pairing and useless as a null control, because those
two columns never read gt_conclusion. Domain bucketing still follows the **evaluated sample's own**
domain (not the partner's), so the buckets line up with the real results table for side-by-side
comparison.

Reuses all the basic functions of ``text_alignment_eval.py`` (ROI coordinate conversion / hit test,
shape judge calls / resumable checkpointing / domain mapping) instead of re-implementing them.
"""
import argparse
import csv
import json
import os

import numpy as np

from chronos_llm.eval.text_alignment_eval import (
    build_openai_client,
    domain_of,
    group_records_by_domain,
    load_predictions,
    load_roi_reference,
    roi_alignment_stats,
    run_shape_judge_batch,
    summarize_roi,
    summarize_shape,
)

DEFAULT_JSONL = (
    "outputs/eval/forecast/forecast_preds.jsonl"
)
DEFAULT_PARQUET = (
    "data/forecast/"
    "mmtr_forecast_corpus.parquet"
)


def build_shuffled_records(pred_rows, ref_df, pairing="derangement", shift=1, seed=0):
    """Pair every sample's gen_text with the partner's gt_conclusion + gt_roi_abs (swapped together);
    domain/id remain the sample's own. Pairing protocol: see `split_text_judge_eval.build_pairing`
    -- **the final protocol is derangement** (uniform random over the pool); `shift` only reproduces
    historical numbers."""
    from chronos_llm.scripts.utils.split_text_judge_eval import build_pairing
    n = len(pred_rows)
    partner = build_pairing(n, list(range(n)), pairing=pairing, shift=shift, seed=seed)
    records = []
    for i in sorted(partner):
        r = pred_rows[i]
        sid = r["id"]
        pj = pred_rows[partner[i]]
        pid = pj["id"]
        if sid not in ref_df.index or pid not in ref_df.index:
            raise KeyError(f"id {sid!r} or {pid!r} not in the parquet reference table")
        own_ref = ref_df.loc[sid]
        partner_ref = ref_df.loc[pid]
        gt_abs_shuffled = (int(partner_ref["roi_start_idx"]), int(partner_ref["roi_end_idx"]))
        records.append({
            "id": sid,
            "dataset_name": str(own_ref["dataset_name"]),
            "domain": domain_of(str(own_ref["dataset_name"])),
            "gen_text": r.get("gen_text") or "",
            "gt_conclusion": pj.get("gt_conclusion") or "",
            "gt_roi_abs": gt_abs_shuffled,
            "paired_with_id": pid,
        })
    return records


def compute_roi_alignment(records):
    out = []
    for rec in records:
        s = roi_alignment_stats(rec["gen_text"], rec["gt_roi_abs"])
        s["id"] = rec["id"]
        s["domain"] = rec["domain"] or "Unknown"
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", default=DEFAULT_JSONL,
                     help="forecast_preds.jsonl produced by TimeOmni-o1 (real generations of the released checkpoint)")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--pairing", choices=["derangement", "random", "shift"],
                    default="derangement",
                    help="null-hypothesis pairing protocol; final=derangement (uniform random over the pool). "
                         "shift only reproduces historical numbers -- it pairs 97.7%% of samples with a same-dataset neighbour, not a random pairing")
    ap.add_argument("--shift", type=int, default=1, help="only used with --pairing shift")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="outputs/eval/timeomni_o1_shuffled_text_alignment")
    ap.add_argument("--skip_shape_judge", action="store_true")
    ap.add_argument("--shape_limit", type=int, default=0)
    ap.add_argument("--judge_model", default="gpt-5.4")
    ap.add_argument("--api_base_url", default="https://api.openai.com/v1")
    ap.add_argument("--max_workers", type=int, default=5)
    args = ap.parse_args()

    pred_rows = load_predictions(args.jsonl)
    ref_df = load_roi_reference(args.parquet)
    records = build_shuffled_records(pred_rows, ref_df, pairing=args.pairing,
                                     shift=args.shift, seed=args.seed)
    desc = f"shift={args.shift}" if args.pairing == "shift" else f"{args.pairing} seed={args.seed}"
    ds = {r["id"]: r["dataset_name"] for r in records}
    same = sum(1 for r in records if ds.get(r["paired_with_id"]) == r["dataset_name"])
    print(f"loaded {len(records)} records ({args.jsonl}), pairing protocol {desc}; "
          f"same-dataset pairing rate {same / len(records):.1%} (lower = closer to truly random).")

    os.makedirs(args.out_dir, exist_ok=True)

    per_sample_roi = compute_roi_alignment(records)
    overall_roi = summarize_roi(per_sample_roi)
    roi_by_id = {s["id"]: s for s in per_sample_roi}
    dom_groups = group_records_by_domain(records)
    by_domain_roi = {dom: summarize_roi([roi_by_id[r["id"]] for r in recs])
                      for dom, recs in dom_groups.items()}
    print(f"\n{'scope':10} {'n':>4} | {'overlap_hit_rate':>17} | {'iou_mean(strict)':>17} | "
          f"{'extract_fail_rate':>18}")
    print("-" * 78)
    for scope, s in [("ALL", overall_roi)] + sorted(by_domain_roi.items()):
        print(f"{scope:10} {s['n']:>4} | {s['overlap_hit_rate']:17.4f} | "
              f"{s['iou_mean']:17.4f} | {s['extract_fail_rate']:18.4f}")

    overall_shape, by_domain_shape, judge_results = None, {}, {}
    subset = records if args.shape_limit <= 0 else records[:args.shape_limit]
    if not args.skip_shape_judge:
        client = build_openai_client(api_base_url=args.api_base_url)
        ckpt = os.path.join(args.out_dir, "shape_judge_raw.jsonl")
        print(f"\ncalling {args.judge_model} for shape similarity scores (shuffled pairing), n={len(subset)}, "
              f"workers={args.max_workers}, checkpoint={ckpt}")
        judge_results = run_shape_judge_batch(
            subset, client, model=args.judge_model, max_workers=args.max_workers, checkpoint_path=ckpt)
        shape_items = [{"score": judge_results.get(r["id"], {}).get("score")} for r in subset]
        overall_shape = summarize_shape(shape_items)
        subset_dom_groups = group_records_by_domain(subset)
        by_domain_shape = {
            dom: summarize_shape([{"score": judge_results.get(r["id"], {}).get("score")} for r in recs])
            for dom, recs in subset_dom_groups.items()
        }
        print(f"\n{'scope':10} {'n':>4} {'n_scored':>9} | {'mean':>6} | {'std':>6} | "
              f"{'judge_fail_rate':>15}")
        print("-" * 66)
        for scope, s in [("ALL", overall_shape)] + sorted(by_domain_shape.items()):
            print(f"{scope:10} {s['n']:>4} {s['n_scored']:>9} | {s['mean']:6.3f} | "
                  f"{s['std']:6.3f} | {s['judge_fail_rate']:15.4f}")

    roi_csv = os.path.join(args.out_dir, "roi_overlap_summary.csv")
    with open(roi_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "n", "overlap_hit_rate", "iou_mean",
                                           "extract_fail_rate", "overlap_hit_rate_given_extracted"])
        w.writeheader()
        w.writerow({"scope": "ALL", **overall_roi})
        for dom in sorted(by_domain_roi):
            w.writerow({"scope": dom, **by_domain_roi[dom]})
    print(f"\nwrote {roi_csv}")

    if overall_shape is not None:
        shape_csv = os.path.join(args.out_dir, "shape_judge_summary.csv")
        with open(shape_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["scope", "n", "n_scored", "judge_fail_rate",
                                               "mean", "std", "dist_1", "dist_2", "dist_3",
                                               "dist_4", "dist_5"])
            w.writeheader()
            for scope, s in [("ALL", overall_shape)] + sorted(by_domain_shape.items()):
                row = {k: v for k, v in s.items() if k != "distribution"}
                row["scope"] = scope
                for k, v in s["distribution"].items():
                    row[f"dist_{k}"] = v
                w.writerow(row)
        print(f"wrote {shape_csv}")

    per_sample_csv = os.path.join(args.out_dir, "per_sample.csv")
    with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "id", "paired_with_id", "dataset_name", "domain", "gt_roi_start", "gt_roi_end",
            "extracted_roi_start", "extracted_roi_end", "overlap_hit", "iou",
            "shape_score", "shape_reason"])
        w.writeheader()
        for rec in records:
            s = roi_by_id[rec["id"]]
            jr = judge_results.get(rec["id"], {})
            ext = s["extracted"]
            w.writerow({
                "id": rec["id"], "paired_with_id": rec["paired_with_id"],
                "dataset_name": rec["dataset_name"], "domain": rec["domain"] or "Unknown",
                "gt_roi_start": rec["gt_roi_abs"][0], "gt_roi_end": rec["gt_roi_abs"][1],
                "extracted_roi_start": ext[0] if ext else "",
                "extracted_roi_end": ext[1] if ext else "",
                "overlap_hit": s["overlap_hit"], "iou": f"{s['iou']:.4f}",
                "shape_score": jr.get("score", ""), "shape_reason": jr.get("reason", ""),
            })
    print(f"wrote {per_sample_csv}")


if __name__ == "__main__":
    main()
