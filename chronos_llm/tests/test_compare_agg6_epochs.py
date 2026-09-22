"""Unit tests for the per-epoch trend table of `compare_agg6_epochs.py` (pure CPU, small synthetic jsonl).
("agg6" denotes the six-source MMTR understanding pool.)

Two things are guarded above all:
1. **The best epoch and the trend direction must not be computed backwards** -- this table directly decides
   which checkpoint becomes the final model.
2. **The degradation list must really catch degrading items** (the qualitative VeriTime diagnosis relies on
   it to be pinned down as numbers).
"""
import json
import os
import tempfile

from chronos_llm.scripts.utils.compare_agg6_epochs import build_table


def _row(task, gt, pred):
    return {"task": task, "ground_truth": f"Answer: {gt}", "generated_text": f"Answer: {pred}"}


def _write(d, name, rows):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))


def _mk_epoch(root, tag, st_correct, vt_correct):
    """Build one epoch's inference directory: ST-Bench T2 gets st_correct/4 right, VeriTime CTU vt_correct/4."""
    d = os.path.join(root, tag)
    _write(d, "stbench_test.jsonl",
           [_row("stbench_entity", "A", "A" if i < st_correct else "B") for i in range(4)])
    _write(d, "veritime_test.jsonl",
           [_row("veritime_CTU", "X", "X" if i < vt_correct else "Y") for i in range(4)])
    return d


def test_best_epoch_and_trend():
    """ST-Bench rises epoch by epoch, VeriTime falls => both the best column and the trend sign must be right."""
    with tempfile.TemporaryDirectory() as root:
        # ep1: ST 25% / VT 100%   ep2: ST 50% / VT 50%   ep3: ST 100% / VT 25%
        specs = [("ep1", _mk_epoch(root, "e1", 1, 4)),
                 ("ep2", _mk_epoch(root, "e2", 2, 2)),
                 ("ep3", _mk_epoch(root, "e3", 4, 1))]
        lines, values, _ = build_table(specs)

        st = values["ST-Bench T2 entity"]
        vt = values["VeriTime CTU"]
        assert [round(v) for v in st] == [25, 50, 100], st
        assert [round(v) for v in vt] == [100, 50, 25], vt

        st_line = [l for l in lines if l.startswith("| ST-Bench T2")][0]
        vt_line = [l for l in lines if l.startswith("| VeriTime CTU")][0]
        # best epoch: ST at ep3, VT at ep1
        assert st_line.rstrip().split("|")[-3].strip() == "ep3", st_line
        assert vt_line.rstrip().split("|")[-3].strip() == "ep1", vt_line
        # trend sign: ST rises +75, VT falls -75 and is flagged as degraded
        assert "+75.00" in st_line, st_line
        assert "-75.00" in vt_line and "⚠️degraded" in vt_line, vt_line
        # the rising row must **not** be flagged (a flipped sign fails here)
        assert "⚠️degraded" not in st_line, st_line
    print("best epoch / trend sign / degradation flag all correct OK")


def test_missing_epoch_columns_are_skipped():
    """When an epoch lacks a dataset (evaluation interrupted half-way), that cell is nan and does not sink the row."""
    with tempfile.TemporaryDirectory() as root:
        full = _mk_epoch(root, "full", 4, 4)
        partial = os.path.join(root, "partial")      # only ST-Bench, no VeriTime
        _write(partial, "stbench_test.jsonl",
               [_row("stbench_entity", "A", "A") for _ in range(4)])

        lines, values, _ = build_table([("ep1", full), ("ep2", partial)])
        assert values["ST-Bench T2 entity"] == [100.0, 100.0]
        vt = values["VeriTime CTU"]
        assert vt[0] == 100.0 and vt[1] != vt[1], f"the absent cell should be nan, got {vt}"
        # absence must not make the row disappear (it should appear as long as the first epoch has a value)
        assert any(l.startswith("| VeriTime CTU") for l in lines)
    print("absent epoch cell is nan and the other columns are unaffected OK")


def test_single_epoch_has_no_bogus_trend():
    """With a single epoch no trend / best must be fabricated (nothing to compare against)."""
    with tempfile.TemporaryDirectory() as root:
        lines, _, _ = build_table([("ep1", _mk_epoch(root, "only", 2, 2))])
        st_line = [l for l in lines if l.startswith("| ST-Bench T2")][0]
        assert st_line.rstrip().endswith("| — | — |"), st_line
        assert "⚠️degraded" not in st_line
    print("single epoch fabricates no trend / best OK")


def test_wrong_task_name_warns(capsys=None):
    """A wrong task name in ROWS must warn -- otherwise the whole row silently disappears and only a
    manual comparison would reveal it.

    Seen in practice: the HiTSR task is lower-case `hitsr_l2` but ROWS said upper-case `hitsr_L2`, so both
    HiTSR rows simply vanished from the table without any hint.
    """
    import io
    import contextlib
    with tempfile.TemporaryDirectory() as root:
        d = os.path.join(root, "e1")
        # the file exists with normal content, but the task name is not one ROWS expects
        _write(d, "hitsr_test.jsonl", [_row("hitsr_WRONGCASE", "A", "A") for _ in range(3)])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            build_table([("ep1", d)])
        msg = err.getvalue()
        assert "HiTSR L2" in msg and "hitsr_WRONGCASE" in msg, \
            f"a task-name mismatch should warn and list the actual task names; stderr={msg!r}"
    print("wrong task name warns and lists the actual values OK")


def test_dir_overlay_prefers_last():
    """`dirA+dirB` overlay: for files present in both, the **later** directory wins; other files still come
    from the earlier one.

    Use case: "re-evaluate a single dataset and overlay it in place" -- e.g. VeriTime had to be re-run with a
    larger max_new_tokens because of truncation, while the 40k ECG rows did not need re-running. Getting the
    overlay direction wrong (taking the earlier one) would silently waste the re-evaluation.
    """
    with tempfile.TemporaryDirectory() as root:
        base = _mk_epoch(root, "base", 1, 1)          # ST 25% / VT 25%
        patch = os.path.join(root, "patch")           # only VeriTime re-evaluated, all correct
        _write(patch, "veritime_test.jsonl",
               [_row("veritime_CTU", "X", "X") for _ in range(4)])

        _, values, _ = build_table([("ep1", f"{base}+{patch}")])
        assert values["VeriTime CTU"] == [100.0], f"VeriTime should come from the overlay directory, got {values['VeriTime CTU']}"
        assert values["ST-Bench T2 entity"] == [25.0], "files absent from the overlay directory should still come from the base directory"
    print("directory overlay prefers the later one, other files unaffected OK")


def test_summary_table_weighting():
    """The three aggregate criteria must really **differ from each other** -- otherwise showing them side by
    side is pointless.

    Build a deliberately conflicting scenario: the large test set (ST-Bench, 40 rows) is best at ep1, the small
    one (VeriTime, 4 rows) at ep2. Count weighting should then pick ep1, the equal-weight mean is pulled
    towards ep2 by the small set, and "number of subtasks where best" is a tie. If all three rows give the same
    best, the weighting is not in effect at all (e.g. counts dropped, degenerating to equal weights).
    """
    from chronos_llm.scripts.utils.compare_agg6_epochs import summary_table
    with tempfile.TemporaryDirectory() as root:
        def mk(tag, st_ok, vt_ok):
            d = os.path.join(root, tag)
            _write(d, "stbench_test.jsonl",
                   [_row("stbench_entity", "A", "A" if i < st_ok else "B") for i in range(40)])
            _write(d, "veritime_test.jsonl",
                   [_row("veritime_CTU", "X", "X" if i < vt_ok else "Y") for i in range(4)])
            return d
        specs = [("ep1", mk("a", 36, 1)), ("ep2", mk("b", 30, 4))]   # ST 90/75, VT 25/100
        _, values, counts = build_table(specs)
        assert counts["ST-Bench T2 entity"] == 40 and counts["VeriTime CTU"] == 4, counts

        out = summary_table(specs, values, counts)
        w = [l for l in out if l.startswith("| Count-weighted mean")][0]
        e = [l for l in out if l.startswith("| Equal-weight mean")][0]
        # count-weighted: (90*40+25*4)/44=84.1 vs (75*40+100*4)/44=77.3 => ep1 wins
        assert w.split("|")[2].strip().startswith("**"), f"count weighting should pick ep1: {w}"
        # equal-weight: (90+25)/2=57.5 vs (75+100)/2=87.5 => ep2 wins (opposite of count weighting)
        assert e.split("|")[3].strip().startswith("**"), f"equal-weight mean should pick ep2 (opposite of count weighting): {e}"
    print("the three aggregate criteria differ (count-weighted picks ep1, equal-weight picks ep2) OK")


if __name__ == "__main__":
    test_summary_table_weighting()
    test_dir_overlay_prefers_last()
    test_wrong_task_name_warns()
    test_best_epoch_and_trend()
    test_missing_epoch_columns_are_skipped()
    test_single_epoch_has_no_bogus_trend()
    print("ALL OK")
