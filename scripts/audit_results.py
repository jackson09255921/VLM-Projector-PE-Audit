#!/usr/bin/env python3
"""Fresh-clone audit of paper numbers, source hashes, and release hygiene.

This audit intentionally uses compact aggregate/provenance artifacts. Raw
benchmark generations, licensed datasets, and 8B checkpoints are not required
and are not redistributed by this repository.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OCR5 = ("ChartQA", "DocVQA", "TextVQA", "InfoVQA", "OCRBench")
SEEDS = ("42", "1234", "2024")


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def close(label: str, actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: {actual} != {expected}")
    print(f"PASS {label}: {actual:.6f}")


def sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def source_method(code: str, sources: dict[str, dict]) -> dict:
    if code == "N":
        return sources["N"]["variants"]["N-r1512"]
    if code == "A":
        return sources["A"]
    if code == "A0":
        return sources["A0"]["variants"]["A0-r1512"]
    return sources["CEF"]["methods"][code]


def source_seed_scores(code: str, source: dict, seed: str) -> tuple[dict[str, float], float, float]:
    row = source["per_seed"][seed]
    scores = row["scores_percent"]
    ocr5 = statistics.fmean(scores[name] for name in OCR5)
    if code == "A":
        all10 = row["ten_benchmark_mean_percent"]
    elif code in {"C", "E", "F"}:
        all10 = row["ten_benchmark_mean_percent"]
    else:
        all10 = row["all10_mean_percent"]
    return scores, ocr5, all10


def audit_sources() -> None:
    manifest = load("provenance/manifest.json")
    for relative, expected in manifest["source_sha256"].items():
        actual = sha256(relative)
        if actual != expected:
            raise AssertionError(f"hash mismatch: {relative}")
        print(f"PASS SHA-256 {relative}")


def audit_results() -> None:
    aggregate = load("results/aggregate_results.json")
    sources = {
        "N": load("results/sources/N.json"),
        "A": load("results/sources/A.json"),
        "A0": load("results/sources/A0.json"),
        "CEF": load("results/sources/CEF.json"),
    }

    for code, reported in aggregate["methods"].items():
        source = source_method(code, sources)
        ocr_values = []
        all_values = []
        for seed in SEEDS:
            _, ocr5, all10 = source_seed_scores(code, source, seed)
            close(f"{code} seed {seed} OCR-5", ocr5, reported["ocr5_per_seed"][seed])
            close(f"{code} seed {seed} All-10", all10, reported["all10_per_seed"][seed])
            ocr_values.append(ocr5)
            all_values.append(all10)
        close(f"{code} OCR-5 mean", statistics.fmean(ocr_values), reported["ocr5_mean"])
        close(f"{code} OCR-5 sample SD", statistics.stdev(ocr_values), reported["ocr5_sample_sd"])
        close(f"{code} All-10 mean", statistics.fmean(all_values), reported["all10_mean"])
        close(f"{code} All-10 sample SD", statistics.stdev(all_values), reported["all10_sample_sd"])

    methods = aggregate["methods"]
    close("PE parameter ratio", methods["A"]["pe_parameters"] / methods["F"]["pe_parameters"], 5.651162790697675)
    close("matched All-10 span", max(x["all10_mean"] for x in methods.values()) - min(x["all10_mean"] for x in methods.values()), 0.1731686677548796)

    paired = [
        methods["F"]["all10_per_seed"][seed] - methods["A0"]["all10_per_seed"][seed]
        for seed in SEEDS
    ]
    if not (paired[0] > 0 and paired[1] > 0 and paired[2] < 0):
        raise AssertionError(f"F-A0 paired signs changed: {paired}")
    close("F-A0 paired mean", statistics.fmean(paired), 0.12116166211367658)


def audit_scheduler() -> None:
    control = load("results/scheduler_control.json")
    gains = [row["C_matched"] - row["C_checkpoint"] for row in control["per_seed"]]
    close("scheduler gain mean", statistics.fmean(gains), control["scheduler_gain_mean_pp"])
    close("scheduler gain sample SD", statistics.stdev(gains), control["scheduler_gain_sample_sd_pp"])
    close("archived LR at step 312", control["old_C_lr_at_step312"], 1.7631058766690842e-5)
    close("matched LR at step 312", control["matched_C_lr_at_step312"], 0.0)


def audit_provenance() -> None:
    manifest = load("provenance/manifest.json")
    primary = manifest["primary_runs"]
    control = manifest["initialization_control_runs"]
    assert primary == {
        "count": 15,
        "methods": ["N", "A", "C", "E", "F"],
        "seeds": [42, 1234, 2024],
        "optimizer_step": 312,
        "scheduler_horizon": 312,
        "final_learning_rate": 0.0,
        "observed_runtime_hours": {"minimum": 10.63, "maximum": 10.71},
    }
    assert control["count"] == 3 and control["optimizer_step"] == 312
    assert control["scheduler_horizon"] == 312 and control["final_learning_rate"] == 0.0
    signatures = manifest["evaluation_identity"]["signatures"]
    evaluation = load("configs/eval/all10.json")["benchmarks"]
    assert set(signatures) == set(evaluation)
    for benchmark, row in signatures.items():
        assert row["examples"] == evaluation[benchmark]["examples"]
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
    print("PASS compact trainer-state and evaluation-identity provenance")


def audit_paper() -> None:
    tex = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    required = (
        "76.879$\\pm$0.323",
        "76.847$\\pm$0.123",
        "76.899$\\pm$0.181",
        "77.020$\\pm$0.072",
        "5.65$\\times$",
        "3.59-point",
        "span only 0.173 points",
    )
    missing = [fragment for fragment in required if fragment not in tex]
    if missing:
        raise AssertionError(f"paper fragments missing: {missing}")
    print("PASS paper headline fragments")


def audit_release_hygiene() -> None:
    forbidden_paths = []
    forbidden_text = []
    token_patterns = (
        re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
        re.compile(rb"hf_[A-Za-z0-9]{20,}"),
        re.compile(rb"(?:sk|sess)-[A-Za-z0-9_-]{20,}"),
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    )
    private_markers = (("/home/" + "fireblue").encode(), ("192.168." + "50.177").encode())
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts or "__pycache__" in relative.parts or not path.is_file():
            continue
        if ".claude" in relative.parts or path.suffix == ".jsonl" or "logs" in relative.parts:
            forbidden_paths.append(str(relative))
        data = path.read_bytes()
        if any(pattern.search(data) for pattern in token_patterns) or any(marker in data for marker in private_markers):
            forbidden_text.append(str(relative))
    if forbidden_paths or forbidden_text:
        raise AssertionError(f"release hygiene failure: paths={forbidden_paths}, text={forbidden_text}")
    print("PASS release hygiene: no raw generations, local settings, private paths, or credential patterns")


def main() -> int:
    audit_sources()
    audit_results()
    audit_scheduler()
    audit_provenance()
    audit_paper()
    audit_release_hygiene()
    print("PASS fresh-clone ICASSP reproducibility audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
