#!/usr/bin/env python3
"""
BATCS — E3: BUDGET SENSITIVITY

Purpose
-------
E3 measures how Random+Budget, UCB+Budget, and BATCS behave as the common
communication budget is varied over four fixed operating points:
    25%, 50%, 75%, 100%.

Frozen research protocol
------------------------
- CIFAR-10
- 50 clients
- exactly 10 Stage-1 clients selected BEFORE local training
- 50 rounds maximum
- 1 local epoch
- batch size 64
- LR 0.01
- SGD momentum 0.9
- Dirichlet alpha 0.5
- server validation size 2000
- seeds = [42, 123, 2024, 7, 2025]
- methods = Random+Budget, UCB+Budget, BATCS

Layer A and Layer B are inherited directly from frozen E1.py.
Layer C is unchanged except for the E3 budget-ratio operating point:
    B0(r) = full-horizon mandatory downlink + r * full-horizon uplink

Actual communication remains:
    B_used = B_downlink + B_preview + B_accepted_full
Rejected full updates are not charged.

Reporting
---------
Two distinct communication metrics are reported:

1) Reference_Envelope_Reduction_pct
   Reduction relative to the full-horizon reference envelope used to define
   the E3 budget operating point.

2) Stage2_vs_UCBBudget_Reduction_pct
   Reduction relative to UCB+Budget:
       (1 - BATCS_MB / UCB+Budget_MB) * 100.
   This is the paper-aligned metric for the controlled Stage-2 comparison.

The experiment artifacts are NOT changed when existing valid job JSONs are
reused. Only aggregate/reporting outputs are regenerated.

Runtime robustness
------------------
Each method × budget ratio is executed in an independent subprocess so a
long-lived Apple MPS runtime cannot contaminate later jobs. Failed jobs are
retried in fresh processes and never enter the final aggregate unless they
pass all integrity checks.

E1.py is frozen and never modified on disk. The worker may temporarily set
E1.MAIN_BUDGET_RATIO in memory to the requested E3 operating point.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# FROZEN E3 DESIGN
# =============================================================================

E3_METHODS = ["Random+Budget", "UCB+Budget", "BATCS"]
E3_BUDGET_RATIOS = [0.25, 0.50, 0.75, 1.00]
E3_SEEDS = [42, 123, 2024, 7, 2025]

# Maximum number of retries after the first attempt.
MAX_RETRIES = 2

# Frozen E1 source hash expected by this E3 protocol.
EXPECTED_E1_SHA256 = (
    "40249a6d3407865692eca22841a7940445642c9661322ab3cbd3f4232a578410"
)

THIS_DIR = Path(__file__).resolve().parent
E1_PATH = THIS_DIR / "E1.py"

RESULT_DIR = THIS_DIR / "batcs_results_final" / "E3_budget_sensitivity"
JOB_DIR = RESULT_DIR / "jobs"
LOG_DIR = RESULT_DIR / "logs"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
JOB_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FROZEN E1 LOADING / VALIDATION
# =============================================================================

def load_e1_module() -> Tuple[Any, str]:
    """Load frozen E1.py without changing it on disk; return module and SHA256."""
    if not E1_PATH.is_file():
        raise FileNotFoundError(
            f"Frozen E1.py not found at: {E1_PATH}\n"
            "Place E1.py beside E3.py."
        )

    source_hash = hashlib.sha256(E1_PATH.read_bytes()).hexdigest()

    spec = importlib.util.spec_from_file_location(
        "batcs_e1_frozen_e3",
        str(E1_PATH),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import frozen E1.py from {E1_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, source_hash


def required_e1_symbols() -> List[str]:
    return [
        "set_seed",
        "load_cifar10",
        "create_validation_split",
        "dirichlet_partition",
        "CIFAR10CNN",
        "clone_state_to_cpu",
        "serialized_size_mb",
        "calculate_budget",
        "run_method",
        "DataLoader",
        "NUM_CLIENTS",
        "CLIENTS_PER_ROUND",
        "NUM_ROUNDS",
        "BATCH_SIZE",
        "SERVER_VALIDATION_SIZE",
        "DIRICHLET_ALPHA",
        "DEVICE",
        "RESULT_DIR",
    ]


def validate_e1(E1: Any, e1_hash: str) -> None:
    """Verify that the loaded E1 is the exact frozen source expected by E3."""
    missing = [name for name in required_e1_symbols() if not hasattr(E1, name)]
    if missing:
        raise AttributeError(
            "Frozen E1.py is missing required symbols; refusing to invent APIs: "
            + ", ".join(missing)
        )

    if e1_hash != EXPECTED_E1_SHA256:
        raise RuntimeError(
            "Frozen E1 SHA256 mismatch.\n"
            f"Observed: {e1_hash}\n"
            f"Expected: {EXPECTED_E1_SHA256}\n"
            "Refusing to run or aggregate E3."
        )

    expected = {
        "NUM_CLIENTS": 50,
        "CLIENTS_PER_ROUND": 10,
        "NUM_ROUNDS": 50,
        "BATCH_SIZE": 64,
        "LEARNING_RATE": 0.01,
        "MOMENTUM": 0.9,
        "DIRICHLET_ALPHA": 0.5,
        "SERVER_VALIDATION_SIZE": 2000,
    }

    for key, value in expected.items():
        actual = getattr(E1, key)
        if actual != value:
            raise RuntimeError(
                f"Frozen E1 mismatch: {key}={actual!r}; "
                f"expected {value!r} for E3."
            )

    frozen_seeds = list(getattr(E1, "SEEDS", []))
    expected_seeds = [42, 123, 2024, 7, 2025]
    if frozen_seeds != expected_seeds:
        raise RuntimeError(
            f"Unexpected E1 seed list: {frozen_seeds!r}. "
            f"E3 requires the frozen five-seed E1 source: {expected_seeds!r}."
        )


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def ratio_tag(ratio: float) -> str:
    return f"{int(round(ratio * 100)):03d}pct"


def job_key(method: str, ratio: float, seed: int) -> str:
    return f"{method}/{ratio:.2f}/seed={seed}"


def job_path(method: str, ratio: float, seed: int) -> Path:
    safe_method = method.replace("+", "plus").replace(" ", "_")
    return JOB_DIR / f"{safe_method}__budget{ratio_tag(ratio)}__seed{seed}.json"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def write_atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    os.replace(tmp, path)


# =============================================================================
# JOB INTEGRITY
# =============================================================================

def check_job_invariants(
    method: str,
    ratio: float,
    seed: int,
    summary: Dict[str, Any],
    rounds: List[Dict[str, Any]],
    budget_mb: float,
    envelope_mb: float,
) -> List[str]:
    problems: List[str] = []

    if summary.get("method") != method:
        problems.append("summary method mismatch")

    if int(summary.get("seed", -1)) != seed:
        problems.append("summary seed mismatch")

    if not math.isclose(
        float(summary.get("budget_ratio", -1)),
        ratio,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        problems.append("summary budget_ratio mismatch")

    if not math.isclose(
        float(summary.get("budget_mb", -1)),
        budget_mb,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        problems.append("summary budget_mb mismatch")

    if not math.isclose(
        float(summary.get("fedavg_envelope_mb", -1)),
        envelope_mb,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        problems.append("summary envelope mismatch")

    accuracy = float(summary.get("final_accuracy", float("nan")))
    if not math.isfinite(accuracy):
        problems.append("final_accuracy is NaN/Inf")

    rounds_completed = int(summary.get("rounds_completed", 0))
    if rounds_completed < 1:
        problems.append("rounds_completed < 1")
    if rounds_completed > 50:
        problems.append("rounds_completed > 50")

    if not rounds:
        problems.append("round_records are empty")

    cumulative_prev = -math.inf

    for row in rounds:
        row_ratio = row.get("Budget_Ratio")
        if row_ratio is not None and not math.isclose(
            float(row_ratio), ratio, rel_tol=0.0, abs_tol=1e-12
        ):
            problems.append(f"round {row.get('Round')}: Budget_Ratio mismatch")

        if row.get("Method") is not None and row.get("Method") != method:
            problems.append(f"round {row.get('Round')}: Method mismatch")

        if row.get("Seed") is not None and int(row.get("Seed")) != seed:
            problems.append(f"round {row.get('Round')}: Seed mismatch")

        selected = int(row.get("Selected", -1))
        accepted = int(row.get("Accepted", -1))

        if selected != 10:
            problems.append(f"round {row.get('Round')}: Selected != 10")

        if accepted < 0 or accepted > 10:
            problems.append(
                f"round {row.get('Round')}: Accepted outside [0,10]"
            )

        downlink = float(row.get("Downlink_MB", 0.0))
        preview = float(row.get("Preview_MB", 0.0))
        accepted_full = float(row.get("Accepted_Full_MB", 0.0))
        round_total = float(
            row.get("Round_Communication_MB", float("nan"))
        )
        cumulative = float(
            row.get("Cumulative_Communication_MB", float("nan"))
        )

        expected_total = downlink + preview + accepted_full

        if (
            not math.isfinite(round_total)
            or abs(round_total - expected_total) > 1e-6
        ):
            problems.append(
                f"round {row.get('Round')}: communication identity violated"
            )

        if not math.isfinite(cumulative):
            problems.append(
                f"round {row.get('Round')}: cumulative communication is NaN/Inf"
            )
        elif cumulative < cumulative_prev - 1e-9:
            problems.append(
                f"round {row.get('Round')}: cumulative communication not monotone"
            )

        if cumulative > budget_mb + 1e-6:
            problems.append(
                f"round {row.get('Round')}: cumulative communication exceeds budget"
            )

        cumulative_prev = cumulative

    if len(rounds) != rounds_completed:
        problems.append(
            f"round count mismatch: {len(rounds)} records vs "
            f"{rounds_completed} completed"
        )

    if rounds:
        ordered_rounds = [int(r.get("Round", -1)) for r in rounds]
        expected_rounds = list(range(1, len(rounds) + 1))
        if ordered_rounds != expected_rounds:
            problems.append(
                f"round numbering is not exactly 1..{len(rounds)}"
            )

        last_acc = float(
            rounds[-1].get("Test_Accuracy_pct", float("nan"))
        )
        if not math.isfinite(last_acc):
            problems.append("final round test accuracy is NaN/Inf")
        elif abs(last_acc - accuracy) > 1e-9:
            problems.append(
                "final_accuracy does not match final round test accuracy"
            )

    if ratio < 0.25 - 1e-12 or ratio > 1.0 + 1e-12:
        problems.append("budget ratio outside E3 range")

    if budget_mb > envelope_mb + 1e-8:
        problems.append("budget exceeds reference envelope")

    return problems


# =============================================================================
# WORKER
# =============================================================================

def run_worker(
    method: str,
    ratio: float,
    seed: int,
    out_path: Path,
) -> None:
    """Run one method × budget × seed job in a fresh process."""
    E1, e1_hash = load_e1_module()
    validate_e1(E1, e1_hash)

    # E3 operating point only. E1 source file remains untouched.
    E1.MAIN_BUDGET_RATIO = float(ratio)
    E1.set_seed(seed)

    print("=" * 100, flush=True)
    print(
        f"E3 WORKER | method={method} | "
        f"budget_ratio={ratio:.2f} | seed={seed}",
        flush=True,
    )
    print(f"PID={os.getpid()} | device={E1.DEVICE}", flush=True)
    print(f"E1 SHA256={e1_hash}", flush=True)
    print("MPS fallback: enabled", flush=True)
    print("=" * 100, flush=True)

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # Exact frozen E1 data pipeline.
    train_set, train_eval_set, test_set = E1.load_cifar10()

    labels = np.asarray(train_set.targets, dtype=np.int64)

    test_loader = E1.DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    validation_indices, client_pool = E1.create_validation_split(
        labels,
        E1.SERVER_VALIDATION_SIZE,
        seed,
    )

    client_indices = E1.dirichlet_partition(
        labels,
        client_pool,
        E1.NUM_CLIENTS,
        E1.DIRICHLET_ALPHA,
        seed,
    )

    measurement_model = E1.CIFAR10CNN()
    measurement_state = E1.clone_state_to_cpu(
        measurement_model.state_dict()
    )
    model_mb = E1.serialized_size_mb(measurement_state)

    budget_mb, envelope_mb = E1.calculate_budget(model_mb)
    E1.fedavg_envelope_mb_global = envelope_mb

    print(
        f"model={model_mb:.6f} MB | "
        f"B0={budget_mb:.6f} MB | "
        f"FedAvg envelope={envelope_mb:.6f} MB",
        flush=True,
    )

    summary, rounds, predictor = E1.run_method(
        method=method,
        seed=seed,
        train_set=train_set,
        train_eval_set=train_eval_set,
        test_loader=test_loader,
        client_indices=client_indices,
        validation_indices=validation_indices,
        model_mb=model_mb,
        budget_mb=budget_mb,
    )

    # Copy E1 output before adding E3 metadata.
    summary = dict(summary)
    summary.update(
        {
            "experiment": "E3_budget_sensitivity",
            "budget_ratio": float(ratio),
            "budget_mb": float(budget_mb),
            "fedavg_envelope_mb": float(envelope_mb),
            "seed": int(seed),
            "method": method,
            "e1_source_sha256": e1_hash,
            "device": str(E1.DEVICE),
        }
    )

    # Preserve the E1 round records numerically, while adding only
    # E3 identifiers needed for unambiguous cross-budget aggregation.
    normalized_rounds: List[Dict[str, Any]] = []
    for raw_row in rounds:
        row = dict(raw_row)
        row["Budget_Ratio"] = float(ratio)
        row["Budget_Limit_MB"] = float(budget_mb)
        row["Method"] = method
        row["Seed"] = int(seed)
        normalized_rounds.append(row)

    problems = check_job_invariants(
        method=method,
        ratio=ratio,
        seed=seed,
        summary=summary,
        rounds=normalized_rounds,
        budget_mb=budget_mb,
        envelope_mb=envelope_mb,
    )

    if problems:
        raise RuntimeError(
            f"E3 invariant failure for {method}/budget={ratio:.2f}/seed={seed}: "
            + "; ".join(problems)
        )

    payload = {
        "experiment": "E3_budget_sensitivity",
        "method": method,
        "budget_ratio": float(ratio),
        "seed": int(seed),
        "summary": summary,
        "round_records": normalized_rounds,
        "predictor_records": predictor,
        "invariants_passed": True,
        "e1_source_sha256": e1_hash,
        "e3_reporting_version": "2.0",
        "stage2_metric_definition": (
            "1 - BATCS_total_MB / UCB+Budget_total_MB"
        ),
    }

    write_atomic_json(out_path, payload)

    print(
        f"[worker] SUCCESS | {method} | budget={ratio:.2f} | seed={seed} | "
        f"acc={summary['final_accuracy']:.2f}% | "
        f"MB={summary['total_mb']:.2f} | "
        f"rounds={summary['rounds_completed']}",
        flush=True,
    )

    # Best-effort cleanup.
    try:
        del train_set, train_eval_set, test_set, test_loader
    except Exception:
        pass

    gc.collect()

    try:
        if (
            hasattr(E1.torch.backends, "mps")
            and E1.torch.backends.mps.is_available()
        ):
            E1.torch.mps.synchronize()
            E1.torch.mps.empty_cache()
    except Exception:
        pass


# =============================================================================
# EXISTING JOB VALIDATION / RESUME
# =============================================================================

def validate_existing(
    path: Path,
    method: str,
    ratio: float,
    seed: int,
) -> bool:
    """Validate an existing job artifact before allowing resume."""
    if not path.is_file():
        return False

    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        if payload.get("method") != method:
            return False

        if int(payload.get("seed", -1)) != seed:
            return False

        if not math.isclose(
            float(payload.get("budget_ratio", -1)),
            ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False

        if payload.get("e1_source_sha256") != EXPECTED_E1_SHA256:
            return False

        if not payload.get("invariants_passed", False):
            return False

        summary = payload["summary"]
        rounds = payload["round_records"]

        problems = check_job_invariants(
            method=method,
            ratio=ratio,
            seed=seed,
            summary=summary,
            rounds=rounds,
            budget_mb=float(summary["budget_mb"]),
            envelope_mb=float(summary["fedavg_envelope_mb"]),
        )

        return not problems

    except Exception:
        return False


# =============================================================================
# LAUNCHER
# =============================================================================

def launch_job(
    method: str,
    ratio: float,
    seed: int,
) -> bool:
    out_path = job_path(method, ratio, seed)

    if validate_existing(out_path, method, ratio, seed):
        print(
            f"[launcher] REUSING valid result: "
            f"{method}/{ratio:.2f}/seed={seed}"
        )
        return True

    safe_method = method.replace("+", "plus").replace(" ", "_")

    for attempt in range(1, MAX_RETRIES + 2):
        log_path = LOG_DIR / (
            f"{safe_method}__budget{ratio_tag(ratio)}__"
            f"seed{seed}__attempt{attempt}.log"
        )

        print(
            f"[launcher] START {method}/{ratio:.2f}/seed={seed} "
            f"attempt {attempt}"
        )

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--method",
            method,
            "--ratio",
            str(ratio),
            "--seed",
            str(seed),
            "--out",
            str(out_path),
        ]

        env = dict(os.environ)
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        try:
            with open(log_path, "w", encoding="utf-8") as log:
                proc = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=False,
                )

            code = int(proc.returncode)

        except Exception as exc:
            print(f"[launcher] subprocess failure: {exc}")
            code = -1

        # IMPORTANT:
        # A worker is successful only when it exits 0 AND its artifact passes
        # the full integrity validator. SystemExit(0) is not interpreted as
        # an error because the worker main block catches only Exception.
        if code == 0 and validate_existing(
            out_path,
            method,
            ratio,
            seed,
        ):
            print(
                f"[launcher] SUCCESS {method}/{ratio:.2f}/seed={seed} "
                f"attempt={attempt}"
            )
            return True

        if code < 0:
            status = f"signal_or_launcher_error={code}"
        else:
            status = f"exit={code}"

        print(
            f"[launcher] FAILED {method}/{ratio:.2f}/seed={seed} "
            f"attempt={attempt} ({status}); see {log_path}"
        )

        if out_path.exists() and not validate_existing(
            out_path,
            method,
            ratio,
            seed,
        ):
            try:
                out_path.unlink()
            except OSError:
                pass

    print(
        f"[launcher] GIVING UP: "
        f"{method}/{ratio:.2f}/seed={seed}"
    )
    return False


# =============================================================================
# REPORTING
# =============================================================================

def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def aggregate_results(
    jobs: List[Tuple[str, float, int]],
) -> None:
    """Regenerate reporting artifacts from validated job JSONs only."""
    payloads = []

    for method, ratio, seed in jobs:
        path = job_path(method, ratio, seed)

        if not validate_existing(path, method, ratio, seed):
            raise RuntimeError(
                f"Refusing to aggregate invalid job artifact: "
                f"{job_key(method, ratio, seed)}"
            )

        with open(path, "r", encoding="utf-8") as fh:
            payloads.append(json.load(fh))

    summaries = [p["summary"] for p in payloads]

    rounds: List[Dict[str, Any]] = []
    predictors: List[Dict[str, Any]] = []

    for p in payloads:
    # Normalize E3 identifiers from the validated job payload itself.
    # This repairs legacy round records that may predate the E3 metadata
    # fields, without changing any experimental measurements.
        for raw_row in p["round_records"]:
            row = dict(raw_row)
            row["Budget_Ratio"] = float(p["budget_ratio"])
            row["Budget_Limit_MB"] = float(p["summary"]["budget_mb"])
            row["Method"] = p["method"]
            row["Seed"] = int(p["seed"])
            rounds.append(row)

        predictors.extend(p.get("predictor_records", []))

    seed_df = pd.DataFrame(summaries)
    round_df = pd.DataFrame(rounds)
    predictor_df = pd.DataFrame(predictors)

    # Add/normalize explicit budget identifiers in round-level data.
    if not round_df.empty:
        if "Budget_Ratio" not in round_df.columns:
            round_df["Budget_Ratio"] = np.nan
        if "Budget_Limit_MB" not in round_df.columns:
            round_df["Budget_Limit_MB"] = np.nan
        if "Method" not in round_df.columns:
            round_df["Method"] = ""
        if "Seed" not in round_df.columns:
            round_df["Seed"] = -1

    # Sort by budget operating point first, then method, seed, round.
    seed_df = seed_df.sort_values(
        ["budget_ratio", "method", "seed"]
    ).reset_index(drop=True)

    if not round_df.empty:
        round_df = round_df.sort_values(
            ["Budget_Ratio", "Method", "Seed", "Round"]
        ).reset_index(drop=True)

    seed_df.to_csv(
        RESULT_DIR / "E3_seed_results.csv",
        index=False,
    )

    round_df.to_csv(
        RESULT_DIR / "E3_round_results.csv",
        index=False,
    )

    if not predictor_df.empty:
        predictor_df.to_csv(
            RESULT_DIR / "E3_BATCS_predictor.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # Paper-aligned E3 table.
    #
    # Reference_Envelope_Reduction_pct:
    #   reduction relative to the reference full-horizon envelope.
    #
    # Stage2_vs_UCBBudget_Reduction_pct:
    #   reduction relative to UCB+Budget at the same budget ratio.
    #   UCB+Budget itself is 0% by definition.
    # -------------------------------------------------------------------------
    table_rows: List[Dict[str, Any]] = []

    for ratio in E3_BUDGET_RATIOS:
        ratio_df = seed_df[
            np.isclose(
                seed_df["budget_ratio"].astype(float),
                ratio,
                atol=1e-12,
            )
        ].copy()

        if ratio_df.empty:
            continue

        ucb_df = ratio_df[
            ratio_df["method"] == "UCB+Budget"
        ].copy()

        ucb_mean_mb = (
            _safe_mean(ucb_df["total_mb"])
            if not ucb_df.empty
            else float("nan")
        )

        for method in E3_METHODS:
            subset = ratio_df[
                ratio_df["method"] == method
            ].copy()

            if subset.empty:
                continue

            acc_mean = _safe_mean(subset["final_accuracy"])
            mb_mean = _safe_mean(subset["total_mb"])
            rounds_mean = _safe_mean(subset["rounds_completed"])
            envelope_mean = _safe_mean(subset["fedavg_envelope_mb"])

            if not math.isfinite(envelope_mean) or envelope_mean <= 0.0:
                reference_reduction = float("nan")
            else:
                reference_reduction = (
                    1.0 - mb_mean / envelope_mean
                ) * 100.0

            if method == "UCB+Budget":
                stage2_reduction = 0.0
            elif method == "BATCS" and math.isfinite(ucb_mean_mb):
                stage2_reduction = (
                    1.0 - mb_mean / ucb_mean_mb
                ) * 100.0
            else:
                stage2_reduction = float("nan")

            acc_values = pd.to_numeric(
                subset["final_accuracy"],
                errors="coerce",
            ).dropna().astype(float)

            mb_values = pd.to_numeric(
                subset["total_mb"],
                errors="coerce",
            ).dropna().astype(float)

            acc_std = (
                float(acc_values.std(ddof=1))
                if len(acc_values) > 1
                else 0.0
            )

            mb_std = (
                float(mb_values.std(ddof=1))
                if len(mb_values) > 1
                else 0.0
            )

            table_rows.append(
                {
                    "Budget_Ratio": ratio,
                    "Method": method,
                    "Accuracy": (
                        f"{acc_mean:.2f} ± {acc_std:.2f}"
                    ),
                    "Total_MB": (
                        f"{mb_mean:.2f} ± {mb_std:.2f}"
                    ),
                    "Rounds_Completed": (
                        f"{rounds_mean:.0f}"
                    ),
                    "Reference_Envelope_Reduction_pct": (
                        f"{reference_reduction:.2f}"
                        if math.isfinite(reference_reduction)
                        else ""
                    ),
                    "Stage2_vs_UCBBudget_Reduction_pct": (
                        f"{stage2_reduction:.2f}"
                        if math.isfinite(stage2_reduction)
                        else ""
                    ),
                    "Accuracy_per_MB": (
                        f"{acc_mean / max(mb_mean, 1e-12):.4f}"
                    ),
                }
            )

    table_df = pd.DataFrame(table_rows)

    table_df.to_csv(
        RESULT_DIR / "E3_Table.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # BATCS vs UCB+Budget pairwise comparison.
    # This is descriptive; E3 uses five paired seeds and no superiority
    # claim is imposed. 
    # -------------------------------------------------------------------------
    comparisons: List[Dict[str, Any]] = []

    for ratio in E3_BUDGET_RATIOS:
        a = seed_df[
            np.isclose(
                seed_df["budget_ratio"].astype(float),
                ratio,
                atol=1e-12,
            )
            & (seed_df["method"] == "BATCS")
        ][["seed", "final_accuracy", "total_mb"]].rename(
            columns={
                "final_accuracy": "batcs_acc",
                "total_mb": "batcs_mb",
            }
        )

        b = seed_df[
            np.isclose(
                seed_df["budget_ratio"].astype(float),
                ratio,
                atol=1e-12,
            )
            & (seed_df["method"] == "UCB+Budget")
        ][["seed", "final_accuracy", "total_mb"]].rename(
            columns={
                "final_accuracy": "ucb_acc",
                "total_mb": "ucb_mb",
            }
        )

        m = a.merge(b, on="seed", how="inner")

        if len(m):
            acc_diffs = (
                m["batcs_acc"].astype(float)
                - m["ucb_acc"].astype(float)
            )
            comm_diffs = (
                m["batcs_mb"].astype(float)
                - m["ucb_mb"].astype(float)
            )

            batcs_mb_mean = float(m["batcs_mb"].mean())
            ucb_mb_mean = float(m["ucb_mb"].mean())

            if ucb_mb_mean > 0.0:
                stage2_reduction = (
                    1.0 - batcs_mb_mean / ucb_mb_mean
                ) * 100.0
            else:
                stage2_reduction = None

            comparisons.append(
                {
                    "budget_ratio": ratio,
                    "paired_seeds": [
                        int(x) for x in m["seed"].tolist()
                    ],
                    "accuracy_difference_mean_pp": float(
                        acc_diffs.mean()
                    ),
                    "accuracy_difference_std_pp": (
                        float(acc_diffs.std(ddof=1))
                        if len(acc_diffs) > 1
                        else 0.0
                    ),
                    "communication_difference_mean_mb": float(
                        comm_diffs.mean()
                    ),
                    "communication_difference_std_mb": (
                        float(comm_diffs.std(ddof=1))
                        if len(comm_diffs) > 1
                        else 0.0
                    ),
                    "stage2_communication_reduction_pct": (
                        float(stage2_reduction)
                        if stage2_reduction is not None
                        else None
                    ),
                    "batcs_mean_accuracy": float(
                        m["batcs_acc"].mean()
                    ),
                    "ucb_budget_mean_accuracy": float(
                        m["ucb_acc"].mean()
                    ),
                    "batcs_mean_communication_mb": batcs_mb_mean,
                    "ucb_budget_mean_communication_mb": ucb_mb_mean,
                    "note": (
                        "Descriptive E3 comparison; five paired seeds; "
                        "no superiority is imposed."
                    ),
                }
            )
        else:
            comparisons.append(
                {
                    "budget_ratio": ratio,
                    "paired_seeds": [],
                    "accuracy_difference_mean_pp": None,
                    "accuracy_difference_std_pp": None,
                    "communication_difference_mean_mb": None,
                    "communication_difference_std_mb": None,
                    "stage2_communication_reduction_pct": None,
                    "batcs_mean_accuracy": None,
                    "ucb_budget_mean_accuracy": None,
                    "batcs_mean_communication_mb": None,
                    "ucb_budget_mean_communication_mb": None,
                    "note": (
                        "No paired BATCS/UCB+Budget observation available."
                    ),
                }
            )

    write_atomic_json(
        RESULT_DIR / "E3_BATCS_vs_UCBBudget_by_budget.json",
        {
            "experiment": "E3_budget_sensitivity",
            "stage2_metric_definition": (
                "(1 - BATCS_total_MB / UCB+Budget_total_MB) * 100"
            ),
            "comparisons": comparisons,
        },
    )


# =============================================================================
# MANIFEST
# =============================================================================

def write_manifest(
    e1_hash: str,
    statuses: Dict[str, bool],
    complete: bool,
) -> None:
    manifest = {
        "experiment": "E3_budget_sensitivity",
        "e1_source_path": str(E1_PATH),
        "e1_source_sha256": e1_hash,
        "expected_e1_source_sha256": EXPECTED_E1_SHA256,
        "e3_methods": E3_METHODS,
        "e3_budget_ratios": E3_BUDGET_RATIOS,
        "e3_seeds": E3_SEEDS,
        "num_jobs": (
            len(E3_METHODS)
            * len(E3_BUDGET_RATIOS)
            * len(E3_SEEDS)
        ),
        "mps_process_isolation": True,
        "max_retries_per_job": MAX_RETRIES,
        "reporting_version": "2.0",
        "stage2_reporting_baseline": "UCB+Budget",
        "reference_envelope_metric_retained": True,
        "job_status": statuses,
        "all_jobs_valid": bool(complete),
        "timestamp_unix": time.time(),
    }

    write_atomic_json(
        RESULT_DIR / "run_manifest.json",
        manifest,
    )


# =============================================================================
# LAUNCHER MAIN
# =============================================================================

def main_launcher() -> int:
    E1, e1_hash = load_e1_module()
    validate_e1(E1, e1_hash)

    print("=" * 100)
    print("BATCS — E3: BUDGET SENSITIVITY")
    print("=" * 100)
    print(f"Frozen E1.py: {E1_PATH}")
    print(f"E1 SHA256: {e1_hash}")
    print(f"Methods: {E3_METHODS}")
    print(f"Budget ratios: {E3_BUDGET_RATIOS}")
    print(f"Seeds: {E3_SEEDS}")
    print("Stage 1: EXACTLY 10/50 before local training")
    print(
        "Layer C: B0 = full-horizon downlink + "
        "ratio × full-horizon uplink"
    )
    print("Stage-2 communication comparison: BATCS vs UCB+Budget")
    print("MPS: each method × budget × seed isolated in its own process")
    print("=" * 100)

    jobs = [
        (method, ratio, seed)
        for ratio in E3_BUDGET_RATIOS
        for method in E3_METHODS
        for seed in E3_SEEDS
    ]

    statuses: Dict[str, bool] = {}

    for method, ratio, seed in jobs:
        key = job_key(method, ratio, seed)
        statuses[key] = launch_job(
            method,
            ratio,
            seed,
        )

    complete = all(statuses.values())

    write_manifest(
        e1_hash=e1_hash,
        statuses=statuses,
        complete=complete,
    )

    if not complete:
        print(
            "\nE3 ABORTED: at least one job is invalid. "
            "Nothing is aggregated."
        )
        for key, ok in statuses.items():
            if not ok:
                print(f"  INVALID: {key}")
        return 1

    # Existing valid jobs can be reused. This regenerates only reporting.
    aggregate_results(jobs)

    print("\n" + "=" * 100)
    print("TABLE — E3 BUDGET SENSITIVITY")
    print("=" * 100)

    table_path = RESULT_DIR / "E3_Table.csv"
    print(
        pd.read_csv(table_path).to_string(index=False)
    )

    print("\nResults saved in:", RESULT_DIR.resolve())
    print("E3 COMPLETED")
    return 0


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BATCS E3 — Budget Sensitivity"
    )

    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run one E3 job in a fresh subprocess.",
    )
    parser.add_argument(
        "--method",
        choices=E3_METHODS,
    )
    parser.add_argument(
        "--ratio",
        type=float,
    )
    parser.add_argument(
        "--seed",
        type=int,
    )
    parser.add_argument(
        "--out",
        type=str,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.worker:
        if (
            args.method is None
            or args.ratio is None
            or args.seed is None
            or args.out is None
        ):
            print(
                "worker requires --method --ratio --seed --out",
                file=sys.stderr,
            )
            sys.exit(2)

        if not any(
            math.isclose(
                args.ratio,
                r,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for r in E3_BUDGET_RATIOS
        ):
            print(
                "Unsupported E3 budget ratio.",
                file=sys.stderr,
            )
            sys.exit(2)

        try:
            run_worker(
                args.method,
                float(args.ratio),
                int(args.seed),
                Path(args.out),
            )
        except Exception:
            # IMPORTANT: catch Exception, NOT BaseException.
            # SystemExit(0) is a BaseException and must not be treated
            # as a worker failure.
            traceback.print_exc()
            sys.exit(1)

        sys.exit(0)

    sys.exit(main_launcher())