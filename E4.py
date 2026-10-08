#!/usr/bin/env python3
"""
BATCS — E4: NON-IID SENSITIVITY

Purpose
-------
E4 isolates the effect of client data heterogeneity by varying ONLY the
Dirichlet partition parameter alpha while keeping the frozen E1 protocol,
Stage-1 selection, Stage-2 control, model/training settings, and communication
budget fixed.

Frozen E4 research protocol
---------------------------
- Dataset: CIFAR-10
- Clients: 50
- Stage 1: exactly 10 clients selected BEFORE local training
- Maximum rounds: 50
- Local epochs: 1
- Batch size: 64
- Learning rate: 0.01
- SGD momentum: 0.9
- Server validation size: 2000
- Methods: UCB+Budget, BATCS
- Dirichlet alpha: 1.0, 0.5, 0.1
- Seeds: [42, 123, 2024, 7, 2025]
- Budget ratio: 0.50 only

Layer A, Layer B, and the communication accounting are inherited directly
from the frozen E1.py. E4 changes ONLY E1.DIRICHLET_ALPHA in memory inside an
isolated worker process; E1.py is never edited on disk.

Budget definition (unchanged from the frozen protocol)
--------------------------------------------------------
    B0 = full-horizon mandatory downlink
         + 0.50 * full-horizon uplink envelope

Actual communication
--------------------
    B_used = B_downlink + B_preview + B_accepted_full

Rejected full updates are not charged.

Reporting
---------
E4 reports:
- Reference_Envelope_Reduction_pct: reduction relative to the full-horizon
  reference envelope used by the fixed 0.50 budget operating point.
- Stage2_vs_UCBBudget_Reduction_pct: BATCS communication reduction relative to
  UCB+Budget at the SAME alpha:
      (1 - BATCS_total_MB / UCB+Budget_total_MB) * 100

Because E4 is a five-seed sensitivity experiment, head-to-head comparisons are
descriptive paired comparisons only. No significance or superiority claim is
generated.

Runtime robustness
------------------
Each method x alpha x seed job runs in an independent subprocess, with bounded
retries, a hard timeout, crash-signal detection, faulthandler logging, atomic
result writes, and best-effort MPS cleanup before normal worker exit. A result
is accepted only if the serialized job artifact passes all integrity checks.
A native segmentation fault is not catchable inside Python; process isolation
ensures a crashed worker cannot corrupt the launcher or silently enter the
scientific aggregate. This improves robustness but cannot mathematically
guarantee that an Apple/PyTorch native component will never crash.
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

# -----------------------------------------------------------------------------
# Runtime-only crash-resilience settings. These do NOT change the BATCS
# scientific protocol. They are applied before frozen E1.py is imported so
# PyTorch/MPS sees them from process startup.
# -----------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTHONFAULTHANDLER", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import pandas as pd


# =============================================================================
# FROZEN E4 DESIGN
# =============================================================================

E4_METHODS = ["UCB+Budget", "BATCS"]
E4_ALPHAS = [1.0, 0.5, 0.1]
E4_SEEDS = [42, 123, 2024, 7, 2025]
E4_BUDGET_RATIO = 0.50

MAX_RETRIES = 2
JOB_TIMEOUT_SECONDS = 3 * 60 * 60
RETRY_BACKOFF_SECONDS = 5
E4_PROTOCOL_VERSION = "2.0"

EXPECTED_E1_SHA256 = (
    "40249a6d3407865692eca22841a7940445642c9661322ab3cbd3f4232a578410"
)

THIS_DIR = Path(__file__).resolve().parent
E1_PATH = THIS_DIR / "E1.py"
RESULT_DIR = THIS_DIR / "batcs_results_final" / "E4_noniid_sensitivity"
JOB_DIR = RESULT_DIR / "jobs"
LOG_DIR = RESULT_DIR / "logs"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
JOB_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FROZEN E1 LOADING / VALIDATION
# =============================================================================

def load_e1_module() -> Tuple[Any, str]:
    """Load frozen E1.py without modifying it; return module and SHA256."""
    if not E1_PATH.is_file():
        raise FileNotFoundError(
            f"Frozen E1.py not found at {E1_PATH}. "
            "Place E1.py beside E4.py."
        )

    source_hash = hashlib.sha256(E1_PATH.read_bytes()).hexdigest()

    spec = importlib.util.spec_from_file_location(
        "batcs_e1_frozen_e4",
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
        "LEARNING_RATE",
        "MOMENTUM",
        "SERVER_VALIDATION_SIZE",
        "DIRICHLET_ALPHA",
        "MAIN_BUDGET_RATIO",
        "SEEDS",
        "DEVICE",
    ]


def validate_e1(E1: Any, e1_hash: str) -> None:
    """Refuse to run unless E1 is the exact frozen source expected by E4."""
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
            "Refusing to run or aggregate E4."
        )

    expected = {
        "NUM_CLIENTS": 50,
        "CLIENTS_PER_ROUND": 10,
        "NUM_ROUNDS": 50,
        "LOCAL_EPOCHS": 1,
        "BATCH_SIZE": 64,
        "LEARNING_RATE": 0.01,
        "MOMENTUM": 0.9,
        "SERVER_VALIDATION_SIZE": 2000,
        "MAIN_BUDGET_RATIO": 0.50,
        "DIRICHLET_ALPHA": 0.5,
    }

    for key, value in expected.items():
        actual = getattr(E1, key)
        if actual != value:
            raise RuntimeError(
                f"Frozen E1 mismatch before E4 mutation: {key}={actual!r}; "
                f"expected {value!r}."
            )

    frozen_seeds = list(getattr(E1, "SEEDS", []))
    expected_seeds = [42, 123, 2024, 7, 2025]
    if frozen_seeds != expected_seeds:
        raise RuntimeError(
            f"Unexpected E1 seed list: {frozen_seeds!r}. "
            f"E4 requires the frozen five-seed E1 source: {expected_seeds!r}."
        )


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def alpha_tag(alpha: float) -> str:
    return f"alpha{alpha:g}".replace(".", "p")


def job_key(method: str, alpha: float, seed: int) -> str:
    return f"{method}/alpha={alpha:g}/seed={seed}"


def job_path(method: str, alpha: float, seed: int) -> Path:
    safe_method = method.replace("+", "plus").replace(" ", "_")
    return JOB_DIR / f"{safe_method}__{alpha_tag(alpha)}__seed{seed}.json"


def write_atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    os.replace(tmp, path)


def current_e4_source_sha256() -> str:
    """Hash the exact E4 source file so stale artifacts cannot be mixed in."""
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


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


# =============================================================================
# JOB INTEGRITY
# =============================================================================

def check_job_invariants(
    method: str,
    alpha: float,
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
        float(summary.get("alpha", float("nan"))),
        alpha,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        problems.append("summary alpha mismatch")

    if not math.isclose(
        float(summary.get("budget_ratio", float("nan"))),
        E4_BUDGET_RATIO,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        problems.append("summary budget_ratio mismatch")

    if not math.isclose(
        float(summary.get("budget_mb", float("nan"))),
        budget_mb,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        problems.append("summary budget_mb mismatch")

    if not math.isclose(
        float(summary.get("fedavg_envelope_mb", float("nan"))),
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
        row_alpha = row.get("Alpha")
        if row_alpha is not None and not math.isclose(
            float(row_alpha), alpha, rel_tol=0.0, abs_tol=1e-12
        ):
            problems.append(f"round {row.get('Round')}: Alpha mismatch")

        if row.get("Method") is not None and row.get("Method") != method:
            problems.append(f"round {row.get('Round')}: Method mismatch")

        if row.get("Seed") is not None and int(row.get("Seed")) != seed:
            problems.append(f"round {row.get('Round')}: Seed mismatch")

        selected = int(row.get("Selected", -1))
        accepted = int(row.get("Accepted", -1))
        if selected != 10:
            problems.append(f"round {row.get('Round')}: Selected != 10")
        if accepted < 0 or accepted > 10:
            problems.append(f"round {row.get('Round')}: Accepted outside [0,10]")

        downlink = float(row.get("Downlink_MB", 0.0))
        preview = float(row.get("Preview_MB", 0.0))
        accepted_full = float(row.get("Accepted_Full_MB", 0.0))
        round_total = float(row.get("Round_Communication_MB", float("nan")))
        cumulative = float(row.get("Cumulative_Communication_MB", float("nan")))

        expected_total = downlink + preview + accepted_full
        if not math.isfinite(round_total) or abs(round_total - expected_total) > 1e-6:
            problems.append(f"round {row.get('Round')}: communication identity violated")

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
            f"round count mismatch: {len(rounds)} records vs {rounds_completed} completed"
        )

    if rounds:
        ordered_rounds = [int(r.get("Round", -1)) for r in rounds]
        expected_rounds = list(range(1, len(rounds) + 1))
        if ordered_rounds != expected_rounds:
            problems.append(
                f"round numbering is not exactly 1..{len(rounds)}"
            )

        last_acc = float(rounds[-1].get("Test_Accuracy_pct", float("nan")))
        if not math.isfinite(last_acc):
            problems.append("final round test accuracy is NaN/Inf")
        elif abs(last_acc - accuracy) > 1e-9:
            problems.append(
                "final_accuracy does not match final round test accuracy"
            )

    if alpha not in E4_ALPHAS:
        problems.append("alpha outside E4 range")
    if budget_mb > envelope_mb + 1e-8:
        problems.append("budget exceeds reference envelope")

    return problems


# =============================================================================
# WORKER
# =============================================================================

def run_worker(method: str, alpha: float, seed: int, out_path: Path) -> None:
    """Run one E4 method x alpha x seed job in a fresh process."""
    try:
        import faulthandler
        faulthandler.enable(all_threads=True)
    except Exception:
        pass

    E1, e1_hash = load_e1_module()
    validate_e1(E1, e1_hash)

    # -------------------------------------------------------------------------
    # E4's ONLY scientific mutation: Dirichlet alpha.
    # Budget ratio remains fixed at the frozen E1/E3 value of 0.50.
    # E1.py itself is never modified on disk.
    # -------------------------------------------------------------------------
    E1.DIRICHLET_ALPHA = float(alpha)
    E1.MAIN_BUDGET_RATIO = E4_BUDGET_RATIO
    E1.set_seed(seed)

    print("=" * 100, flush=True)
    print(
        f"E4 WORKER | method={method} | alpha={alpha:g} | seed={seed} | "
        f"budget_ratio={E4_BUDGET_RATIO:.2f}",
        flush=True,
    )
    print(f"PID={os.getpid()} | device={E1.DEVICE}", flush=True)
    print(f"E1 SHA256={e1_hash}", flush=True)
    print("E4 mutation: DIRICHLET_ALPHA only", flush=True)
    print("MPS fallback: enabled", flush=True)
    print("faulthandler: enabled where supported", flush=True)
    print("=" * 100, flush=True)

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

    # Explicitly assert that the E4 partition was constructed with the
    # requested alpha through the frozen E1 API rather than silently falling
    # back to MAIN_ALPHA.
    if not isinstance(alpha, float):
        raise TypeError("alpha must be a float")

    measurement_model = E1.CIFAR10CNN()
    measurement_state = E1.clone_state_to_cpu(measurement_model.state_dict())
    model_mb = E1.serialized_size_mb(measurement_state)

    budget_mb, envelope_mb = E1.calculate_budget(model_mb)
    if not math.isclose(
        budget_mb,
        E4_BUDGET_RATIO * envelope_mb + (1.0 - E4_BUDGET_RATIO) *
        (E1.CLIENTS_PER_ROUND * E1.NUM_ROUNDS * model_mb),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        # The expression above is the equivalent form of:
        # downlink_full_horizon + ratio * uplink_full_horizon.
        raise RuntimeError("Frozen E1 budget equation mismatch.")

    E1.fedavg_envelope_mb_global = envelope_mb

    print(
        f"model={model_mb:.6f} MB | B0={budget_mb:.6f} MB | "
        f"FedAvg envelope={envelope_mb:.6f} MB | alpha={alpha:g}",
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

    summary = dict(summary)
    summary.update(
        {
            "experiment": "E4_noniid_sensitivity",
            "alpha": float(alpha),
            "budget_ratio": E4_BUDGET_RATIO,
            "budget_mb": float(budget_mb),
            "fedavg_envelope_mb": float(envelope_mb),
            "seed": int(seed),
            "method": method,
            "e1_source_sha256": e1_hash,
            "device": str(E1.DEVICE),
        }
    )

    normalized_rounds: List[Dict[str, Any]] = []
    for raw_row in rounds:
        row = dict(raw_row)
        row["Alpha"] = float(alpha)
        row["Budget_Ratio"] = E4_BUDGET_RATIO
        row["Budget_Limit_MB"] = float(budget_mb)
        row["Method"] = method
        row["Seed"] = int(seed)
        normalized_rounds.append(row)

    problems = check_job_invariants(
        method=method,
        alpha=alpha,
        seed=seed,
        summary=summary,
        rounds=normalized_rounds,
        budget_mb=budget_mb,
        envelope_mb=envelope_mb,
    )
    if problems:
        raise RuntimeError(
            f"E4 invariant failure for {method}/alpha={alpha:g}/seed={seed}: "
            + "; ".join(problems)
        )

    payload = {
        "experiment": "E4_noniid_sensitivity",
        "method": method,
        "alpha": float(alpha),
        "budget_ratio": E4_BUDGET_RATIO,
        "seed": int(seed),
        "summary": summary,
        "round_records": normalized_rounds,
        "predictor_records": predictor,
        "invariants_passed": True,
        "e1_source_sha256": e1_hash,
        "e4_protocol_version": E4_PROTOCOL_VERSION,
        "e4_source_sha256": current_e4_source_sha256(),
        "stage2_metric_definition": (
            "(1 - BATCS_total_MB / UCB+Budget_total_MB) * 100"
        ),
    }

    write_atomic_json(out_path, payload)

    print(
        f"[worker] SUCCESS | {method} | alpha={alpha:g} | seed={seed} | "
        f"acc={summary['final_accuracy']:.2f}% | "
        f"MB={summary['total_mb']:.2f} | "
        f"rounds={summary['rounds_completed']}",
        flush=True,
    )

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

def validate_existing(path: Path, method: str, alpha: float, seed: int) -> bool:
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
            float(payload.get("alpha", float("nan"))),
            alpha,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
        if not math.isclose(
            float(payload.get("budget_ratio", float("nan"))),
            E4_BUDGET_RATIO,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
        if payload.get("e1_source_sha256") != EXPECTED_E1_SHA256:
            return False
        if payload.get("e4_protocol_version") != E4_PROTOCOL_VERSION:
            return False
        if payload.get("e4_source_sha256") != current_e4_source_sha256():
            return False
        if not payload.get("invariants_passed", False):
            return False

        summary = payload["summary"]
        rounds = payload["round_records"]

        problems = check_job_invariants(
            method=method,
            alpha=alpha,
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

def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Best-effort termination of a timed-out/crashed worker process group."""
    try:
        if proc.poll() is None:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, 15)
                time.sleep(1.0)
            if proc.poll() is None and hasattr(os, "killpg"):
                os.killpg(proc.pid, 9)
            elif proc.poll() is None:
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _tail_log(path: Path, n: int = 25) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return "<unable to read worker log>"


def launch_job(method: str, alpha: float, seed: int) -> bool:
    """Run one job in a fresh process with crash isolation and bounded retries."""
    out_path = job_path(method, alpha, seed)

    if validate_existing(out_path, method, alpha, seed):
        print(
            f"[launcher] REUSING valid result: "
            f"{method}/alpha={alpha:g}/seed={seed}",
            flush=True,
        )
        return True

    safe_method = method.replace("+", "plus").replace(" ", "_")

    for attempt in range(1, MAX_RETRIES + 2):
        log_path = LOG_DIR / (
            f"{safe_method}__{alpha_tag(alpha)}__seed{seed}__attempt{attempt}.log"
        )
        print(
            f"[launcher] START {method}/alpha={alpha:g}/seed={seed} "
            f"attempt {attempt}/{MAX_RETRIES + 1}",
            flush=True,
        )

        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--method", method,
            "--alpha", str(alpha),
            "--seed", str(seed),
            "--out", str(out_path),
        ]
        env = dict(os.environ)
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["PYTHONFAULTHANDLER"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        # Effective reproducibility control at interpreter start.
        env["PYTHONHASHSEED"] = str(seed)

        proc: subprocess.Popen | None = None
        code = -1
        timed_out = False

        try:
            with open(log_path, "w", encoding="utf-8", buffering=1) as log:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                try:
                    proc.wait(timeout=JOB_TIMEOUT_SECONDS)
                    code = int(proc.returncode)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_process_group(proc)
                    proc.wait(timeout=10)
                    code = int(proc.returncode if proc.returncode is not None else -9)
        except Exception as exc:
            print(f"[launcher] subprocess orchestration failure: {exc}", flush=True)
            if proc is not None:
                _terminate_process_group(proc)
            code = -1

        valid = code == 0 and validate_existing(out_path, method, alpha, seed)
        if valid:
            print(
                f"[launcher] SUCCESS {method}/alpha={alpha:g}/seed={seed} "
                f"attempt={attempt}",
                flush=True,
            )
            return True

        if timed_out:
            status = f"TIMEOUT after {JOB_TIMEOUT_SECONDS}s"
        elif code < 0:
            signal_num = -code
            if signal_num == 11:
                status = "SEGMENTATION_FAULT(signal=11)"
            else:
                status = f"NATIVE_SIGNAL(signal={signal_num})"
        else:
            status = f"EXIT_CODE={code}"

        print(
            f"[launcher] FAILED {method}/alpha={alpha:g}/seed={seed} "
            f"attempt={attempt}: {status}",
            flush=True,
        )
        print("[launcher] ---- worker log tail ----", flush=True)
        tail = _tail_log(log_path)
        if tail:
            print(tail, flush=True)
        print("[launcher] --------------------------", flush=True)

        # Never trust a partially written or stale artifact.
        if out_path.exists() and not validate_existing(out_path, method, alpha, seed):
            try:
                out_path.unlink()
            except OSError:
                pass

        if attempt < MAX_RETRIES + 1:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    print(
        f"[launcher] GIVING UP: {method}/alpha={alpha:g}/seed={seed}",
        flush=True,
    )
    return False


# =============================================================================
# REPORTING
# =============================================================================

def _mean_std(values: pd.Series) -> Tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(x) == 0:
        return float("nan"), float("nan")
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return mean, std


def aggregate_results(jobs: List[Tuple[str, float, int]]) -> None:
    payloads = []
    for method, alpha, seed in jobs:
        path = job_path(method, alpha, seed)
        if not validate_existing(path, method, alpha, seed):
            raise RuntimeError(
                f"Refusing to aggregate invalid job artifact: "
                f"{job_key(method, alpha, seed)}"
            )
        with open(path, "r", encoding="utf-8") as fh:
            payloads.append(json.load(fh))

    seed_df = pd.DataFrame([p["summary"] for p in payloads])

    rounds = []
    predictors = []
    for p in payloads:
        for raw_row in p["round_records"]:
            row = dict(raw_row)
            row["Alpha"] = float(p["alpha"])
            row["Budget_Ratio"] = float(p["budget_ratio"])
            row["Budget_Limit_MB"] = float(p["summary"]["budget_mb"])
            row["Method"] = p["method"]
            row["Seed"] = int(p["seed"])
            rounds.append(row)
        for raw_pred in p.get("predictor_records", []):
            pred = dict(raw_pred)
            pred["Alpha"] = float(p["alpha"])
            pred["Budget_Ratio"] = float(p["budget_ratio"])
            pred["Method"] = p["method"]
            pred["Seed"] = int(p["seed"])
            predictors.append(pred)
    round_df = pd.DataFrame(rounds)
    predictor_df = pd.DataFrame(predictors)

    seed_df = seed_df.sort_values(["alpha", "method", "seed"]).reset_index(drop=True)
    if not round_df.empty:
        round_df = round_df.sort_values(
            ["Alpha", "Method", "Seed", "Round"]
        ).reset_index(drop=True)

    seed_df.to_csv(RESULT_DIR / "E4_seed_results.csv", index=False)
    round_df.to_csv(RESULT_DIR / "E4_round_results.csv", index=False)
    if not predictor_df.empty:
        predictor_df.to_csv(RESULT_DIR / "E4_BATCS_predictor.csv", index=False)

    table_rows: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []

    for alpha in E4_ALPHAS:
        alpha_df = seed_df[np.isclose(seed_df["alpha"].astype(float), alpha)].copy()
        ucb_df = alpha_df[alpha_df["method"] == "UCB+Budget"].copy()
        batcs_df = alpha_df[alpha_df["method"] == "BATCS"].copy()

        ucb_mean_mb = float(ucb_df["total_mb"].mean()) if not ucb_df.empty else float("nan")
        batcs_mean_mb = float(batcs_df["total_mb"].mean()) if not batcs_df.empty else float("nan")
        stage2_reduction = (
            (1.0 - batcs_mean_mb / ucb_mean_mb) * 100.0
            if math.isfinite(ucb_mean_mb) and ucb_mean_mb > 0 and math.isfinite(batcs_mean_mb)
            else float("nan")
        )

        if not alpha_df.empty:
            envelope = float(alpha_df["fedavg_envelope_mb"].iloc[0])
        else:
            envelope = float("nan")

        for method in E4_METHODS:
            subset = alpha_df[alpha_df["method"] == method].copy()
            if subset.empty:
                continue

            acc_mean, acc_std = _mean_std(subset["final_accuracy"])
            mb_mean, mb_std = _mean_std(subset["total_mb"])
            rounds_mean = float(pd.to_numeric(subset["rounds_completed"], errors="coerce").mean())
            reference_reduction = (
                (1.0 - mb_mean / envelope) * 100.0
                if math.isfinite(envelope) and envelope > 0
                else float("nan")
            )
            stage2_value = 0.0 if method == "UCB+Budget" else stage2_reduction

            table_rows.append(
                {
                    "Dirichlet_Alpha": alpha,
                    "Method": method,
                    "Accuracy": f"{acc_mean:.2f} ± {acc_std:.2f}",
                    "Total_MB": f"{mb_mean:.2f} ± {mb_std:.2f}",
                    "Rounds_Completed": f"{rounds_mean:.0f}",
                    "Reference_Envelope_Reduction_pct": (
                        f"{reference_reduction:.2f}"
                        if math.isfinite(reference_reduction)
                        else ""
                    ),
                    "Stage2_vs_UCBBudget_Reduction_pct": (
                        f"{stage2_value:.2f}"
                        if math.isfinite(stage2_value)
                        else ""
                    ),
                    "Accuracy_per_MB": (
                        f"{acc_mean / max(mb_mean, 1e-12):.4f}"
                    ),
                }
            )

        if not ucb_df.empty and not batcs_df.empty:
            merged = batcs_df[
                ["seed", "final_accuracy", "total_mb"]
            ].merge(
                ucb_df[["seed", "final_accuracy", "total_mb"]],
                on="seed",
                suffixes=("_batcs", "_ucb"),
            )
            acc_diff = merged["final_accuracy_batcs"] - merged["final_accuracy_ucb"]
            comm_diff = merged["total_mb_batcs"] - merged["total_mb_ucb"]
            comparisons.append(
                {
                    "alpha": alpha,
                    "paired_seeds": [int(x) for x in merged["seed"].tolist()],
                    "accuracy_difference_mean_pp": float(acc_diff.mean()),
                    "accuracy_difference_std_pp": float(acc_diff.std(ddof=1)) if len(acc_diff) > 1 else 0.0,
                    "communication_difference_mean_mb": float(comm_diff.mean()),
                    "communication_difference_std_mb": float(comm_diff.std(ddof=1)) if len(comm_diff) > 1 else 0.0,
                    "stage2_communication_reduction_pct": float(stage2_reduction),
                    "batcs_mean_accuracy": float(merged["final_accuracy_batcs"].mean()),
                    "ucb_budget_mean_accuracy": float(merged["final_accuracy_ucb"].mean()),
                    "batcs_mean_communication_mb": float(merged["total_mb_batcs"].mean()),
                    "ucb_budget_mean_communication_mb": float(merged["total_mb_ucb"].mean()),
                    "note": "Descriptive E4 comparison; five paired seeds; no superiority is imposed.",
                }
            )

    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(RESULT_DIR / "E4_Table.csv", index=False)

    write_atomic_json(
        RESULT_DIR / "E4_BATCS_vs_UCBBudget_by_alpha.json",
        {
            "experiment": "E4_noniid_sensitivity",
            "stage2_metric_definition": "(1 - BATCS_total_MB / UCB+Budget_total_MB) * 100",
            "fixed_budget_ratio": E4_BUDGET_RATIO,
            "comparisons": comparisons,
        },
    )


# =============================================================================
# MANIFEST
# =============================================================================

def write_manifest(e1_hash: str, statuses: Dict[str, bool], complete: bool) -> None:
    write_atomic_json(
        RESULT_DIR / "run_manifest.json",
        {
            "experiment": "E4_noniid_sensitivity",
            "e1_source_path": str(E1_PATH),
            "e1_source_sha256": e1_hash,
            "expected_e1_source_sha256": EXPECTED_E1_SHA256,
            "e4_methods": E4_METHODS,
            "e4_alphas": E4_ALPHAS,
            "e4_seeds": E4_SEEDS,
            "e4_protocol_version": E4_PROTOCOL_VERSION,
            "e4_source_sha256": current_e4_source_sha256(),
            "fixed_budget_ratio": E4_BUDGET_RATIO,
            "num_jobs": len(E4_METHODS) * len(E4_ALPHAS) * len(E4_SEEDS),
            "mps_process_isolation": True,
            "max_retries_per_job": MAX_RETRIES,
            "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
            "retry_backoff_seconds": RETRY_BACKOFF_SECONDS,
            "only_scientific_variable": "Dirichlet alpha",
            "stage2_reporting_baseline": "UCB+Budget",
            "job_status": statuses,
            "all_jobs_valid": bool(complete),
            "timestamp_unix": time.time(),
        },
    )


# =============================================================================
# MAIN LAUNCHER
# =============================================================================

def main_launcher() -> int:
    E1, e1_hash = load_e1_module()
    validate_e1(E1, e1_hash)

    print("=" * 100)
    print("BATCS — E4: NON-IID SENSITIVITY")
    print("=" * 100)
    print(f"Frozen E1.py: {E1_PATH}")
    print(f"E1 SHA256: {e1_hash}")
    print(f"Methods: {E4_METHODS}")
    print(f"Dirichlet alphas: {E4_ALPHAS}")
    print(f"Seed(s): {E4_SEEDS}")
    print(f"Fixed budget ratio: {E4_BUDGET_RATIO:.2f}")
    print("Stage 1: EXACTLY 10/50 before local training")
    print("Only scientific variable: Dirichlet alpha")
    print("Stage-2 communication comparison: BATCS vs UCB+Budget")
    print("MPS: each method × alpha × seed isolated in its own process")
    print(f"Job timeout: {JOB_TIMEOUT_SECONDS}s | retries: {MAX_RETRIES} | protocol: {E4_PROTOCOL_VERSION}")
    print("=" * 100)

    jobs = [
        (method, alpha, seed)
        for alpha in E4_ALPHAS
        for method in E4_METHODS
        for seed in E4_SEEDS
    ]

    statuses: Dict[str, bool] = {}
    for method, alpha, seed in jobs:
        statuses[job_key(method, alpha, seed)] = launch_job(method, alpha, seed)

    complete = all(statuses.values())
    write_manifest(e1_hash, statuses, complete)

    if not complete:
        print("\nE4 ABORTED: at least one job is invalid. Nothing is aggregated.")
        for key, ok in statuses.items():
            if not ok:
                print(f"  INVALID: {key}")
        return 1

    aggregate_results(jobs)

    print("\n" + "=" * 100)
    print("TABLE — E4 NON-IID SENSITIVITY")
    print("=" * 100)
    print(pd.read_csv(RESULT_DIR / "E4_Table.csv").to_string(index=False))
    print("\nResults saved in:", RESULT_DIR.resolve())
    print("E4 COMPLETED")
    return 0


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BATCS E4 — Non-IID Sensitivity")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=E4_METHODS)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.worker:
        if args.method is None or args.alpha is None or args.seed is None or args.out is None:
            print("worker requires --method --alpha --seed --out", file=sys.stderr)
            sys.exit(2)

        if not any(
            math.isclose(args.alpha, a, rel_tol=0.0, abs_tol=1e-12)
            for a in E4_ALPHAS
        ):
            print("Unsupported E4 alpha.", file=sys.stderr)
            sys.exit(2)

        if args.seed not in E4_SEEDS:
            print("Unsupported E4 seed.", file=sys.stderr)
            sys.exit(2)

        try:
            run_worker(
                args.method,
                float(args.alpha),
                int(args.seed),
                Path(args.out),
            )
        except Exception:
            # Catch Exception, NOT BaseException, so SystemExit(0) is never
            # converted into a false scientific failure.
            traceback.print_exc()
            sys.exit(1)

        sys.exit(0)

    sys.exit(main_launcher())