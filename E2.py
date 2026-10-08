#!/usr/bin/env python3
"""
BATCS — E2: STAGE-2 NOVELTY ISOLATION
=======================================================================

RESEARCH QUESTION
    Does the proposed Stage-2 mechanism add value beyond UCB Stage-1
    selection + the common communication budget, and does the learned
    predictor add value beyond a non-learned Stage-2 policy?

E2 METHODS (only these three):
    1. UCB+Budget         -- E1's own implementation, called unmodified.
    2. BATCS-NoPredictor  -- Layer-B ablation: identical Stage-2 pipeline
                             (percentile boundary, borderline pool,
                             dynamic acceptance count, exact subset
                             optimization, post-decision feedback) but
                             the LEARNED ridge predictor is replaced by a
                             non-learned proxy that reports the
                             pre-decision UCB score as the "utility"
                             signal and a fixed 0.0 uncertainty. See
                             UCBOnlyProxyPredictor below.
    3. BATCS              -- E1's own implementation, called unmodified.

THIS FILE DOES NOT MODIFY E1.py. E1.py IS TREATED AS FROZEN, READ-ONLY
SOURCE OF TRUTH. Every actual computation (model, optimizer, aggregation,
serialization, budget definition, Stage-1 UCB, Stage-2 percentile/dynamic-
count/subset-optimization machinery, evaluation) is performed by literally
calling E1's own functions/classes, imported from the E1.py file that must
sit next to this one. E2.py supplies only:
    (a) a duck-typed non-learned predictor stand-in for the ablation,
    (b) a thin orchestration function for the ablation (E1's run_method()
        does not accept a pluggable predictor -- see the "E1 MISMATCHES"
        note printed at start-up and in the accompanying chat message),
    (c) the subprocess launcher/retry/invariant-check/aggregation harness
        requested for MPS-crash isolation.

=======================================================================
IMPORTANT, EXPLICITLY FLAGGED SPEC CONFLICT (read before trusting output)
=======================================================================
The E2 instructions require, simultaneously:
  KEEP (for BATCS-NoPredictor): "same percentile evidence boundary, same
      dynamic acceptance-count mechanism, same borderline pool, same
      exact budget-feasible subset optimization" -- i.e. E1's
      choose_batcs_stage2() must actually reach that code path.
  INVARIANT #9: "predictor sample count = 0 for BATCS-NoPredictor".

Inspecting E1.py's actual choose_batcs_stage2() shows these cannot both
hold: it gates on `predictor.n_samples < BATCS_MIN_PREDICTOR_SAMPLES`
(currently 40) to decide whether to stay in "ucb_protected_warmup" mode
or ever reach the percentile/dynamic-count/borderline/exact-subset
pipeline at all. If the proxy predictor's `n_samples` is pinned at 0
forever (literal invariant #9), E1's unmodified Stage-2 function would
run in warmup mode for all 50 rounds and NEVER exercise the mechanisms
the KEEP list demands. There is no value of that counter that satisfies
both requirements against E1's real, frozen code.

Resolution implemented here (NOPREDICTOR_MATURITY_MODE = "grow", see
below): the proxy's `n_samples` is allowed to advance with the exact
same cadence as the real predictor (+1 per accepted client per round,
via `.update()`), so E1's own warmup/confidence gating opens up and the
KEPT mechanisms actually run. NO LEARNED MODEL EVER EXISTS: the proxy
holds zero coefficients, fits nothing, and its `.predict_with_uncertainty`
ignores the feature vector entirely, returning only the already-computed
pre-decision UCB score. In every output file this is reported under the
literal, unambiguous field `Learned_Ridge_Coefficients_Fitted = False`
/ `Learned_Ridge_Samples = 0` (always) -- THIS is the faithful reading of
"predictor must genuinely be absent". The `Predictor_Samples` column
(kept for schema parity with E1's BATCS output) instead reports the
maturity counter under the explicit alias `Maturity_Counter_NOT_Learning`.

If you require the OTHER reading (literal `n_samples == 0` forever, and
therefore permanent warmup mode, and therefore NOT exercising the KEEP
list), flip NOPREDICTOR_MATURITY_MODE to "pinned_zero" below -- no other
code changes are needed. Nothing here was chosen to make BATCS win or
lose; this is a mechanical consequence of E1's real gating logic and is
documented, not silently picked.
=======================================================================
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
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# E2 FROZEN SETTINGS
# =============================================================================

# FLAG (see "IMPORTANT, EXPLICITLY FLAGGED SPEC CONFLICT" above).
NOPREDICTOR_MATURITY_MODE = "grow"  # controlled maturity clock; proxy never learns

# MISMATCH #1 (disclosed, not silently resolved): the attached E1.py's own
# SEEDS constant is [42, 123, 2024, 7, 2025] (5 seeds), but the E2 spec
# explicitly and repeatedly states seeds=[42,123,2024]. E2 has its own
# frozen settings section naming these three seeds explicitly, so E2 uses
# its OWN seed list below rather than E1.SEEDS. E1.py itself is not
# touched or altered.
E2_SEEDS = [42, 123, 2024, 7, 2025]

E2_METHODS = ["UCB+Budget", "BATCS-NoPredictor", "BATCS"]

MAX_RETRIES = 2  # bounded retry count per job, in a FRESH subprocess each time

THIS_DIR = Path(__file__).resolve().parent
E1_PATH = THIS_DIR / "E1.py"

RESULT_DIR = THIS_DIR / "batcs_results_final" / "E2_stage2_novelty_isolation"
JOB_DIR = RESULT_DIR / "jobs"
LOG_DIR = RESULT_DIR / "logs"


# =============================================================================
# E1 IMPORT (E1.py is frozen; imported, never edited, never reimplemented)
# =============================================================================

def load_e1_module():
    """Dynamically import E1.py from disk (must sit next to this file).
    Returns (module, sha256_hex) so the run manifest can record exactly
    which frozen E1 source produced these results."""
    if not E1_PATH.is_file():
        raise FileNotFoundError(
            f"E1.py not found at {E1_PATH}. E2.py requires the frozen "
            f"E1.py to sit in the same directory so it can import and "
            f"reuse E1's actual functions/classes (per the 'do not "
            f"invent missing APIs' rule)."
        )

    source_bytes = E1_PATH.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()

    spec = importlib.util.spec_from_file_location("E1_frozen", str(E1_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules["E1_frozen"] = module
    spec.loader.exec_module(module)  # runs E1.py at import time; E1.py's
    # own `if __name__ == "__main__": main()` guard means this does NOT
    # execute E1's main() experiment -- only definitions run.

    return module, source_hash


# =============================================================================
# NON-LEARNED STAGE-2 UTILITY PROXY (BATCS-NoPredictor ablation)
# =============================================================================

class UCBOnlyProxyPredictor:
    """Duck-typed stand-in for E1's OnlineRidgePredictor, satisfying the
    EXACT interface E1's choose_batcs_stage2() requires
    (predict_with_uncertainty(x), .n_samples, .update(x, target)) so that
    E1's own unmodified Stage-2 function can be called without any change
    to E1.py.

    NO LEARNING EVER HAPPENS: no coefficients, no matrix, no residual
    tracking. `predict_with_uncertainty` ignores the feature vector `x`
    completely -- the whole point of the ablation is that no function of
    the preview is learned -- and returns the pre-decision UCB score
    already computed for that candidate this round (set via
    `set_round_scores()` before Stage 2 runs each round), with a fixed
    uncertainty of 0.0 (a non-learned proxy has no learned uncertainty
    estimate to report).

    `n_samples` either advances with the same cadence as the real
    predictor (mode="grow", default) or stays pinned at 0 forever
    (mode="pinned_zero") -- see the module docstring's flagged conflict.
    `Learned_Ridge_Samples` always reports 0: literally nothing is ever
    fit, regardless of mode.
    """

    def __init__(self, mode: str = "grow"):
        if mode not in ("grow", "pinned_zero"):
            raise ValueError(f"Unknown NOPREDICTOR_MATURITY_MODE: {mode}")
        self._mode = mode
        self.n_samples = 0
        self.learned_ridge_samples = 0  # literally never incremented
        self._round_scores: Optional[np.ndarray] = None
        self._call_index = 0

    def set_round_scores(self, ucb_scores_in_feature_order) -> None:
        """Must be called once per round, before Stage 2, with the
        pre-decision UCB scores in the SAME order as the `features` list
        passed to choose_batcs_stage2() (E1 guarantees this order is the
        Stage-1 `selected` order; E2's orchestration preserves it)."""
        self._round_scores = np.asarray(ucb_scores_in_feature_order, dtype=np.float64)
        self._call_index = 0

    def predict_with_uncertainty(self, x) -> Tuple[float, float]:
        del x  # intentionally unused: no function of the preview is learned
        if self._round_scores is None:
            raise RuntimeError(
                "UCBOnlyProxyPredictor.set_round_scores() must be called "
                "before Stage 2 each round."
            )
        if self._call_index >= len(self._round_scores):
            raise RuntimeError(
                "UCBOnlyProxyPredictor received more predict calls than "
                "candidates this round; ordering assumption violated."
            )
        score = float(self._round_scores[self._call_index])
        self._call_index += 1
        return score, 0.0

    def update(self, x, target) -> float:
        del x, target  # intentionally unused: genuinely no learning occurs
        if self._mode == "grow":
            self.n_samples += 1
        return 0.0


# =============================================================================
# BATCS-NoPredictor ORCHESTRATION
#
# E1.py has no standalone, reusable "BATCS orchestration" function --
# run_method() hardcodes per-method branches inline (if method == "BATCS":
# ... elif method == "UCB+Budget": ...), so there is no parameter to swap
# in a different predictor for a new method string. This function mirrors
# run_method()'s BATCS branch structure step-for-step so the ablation is
# scientifically faithful, but every actual computation below is a direct,
# unmodified call into E1's own functions/classes -- nothing is
# reimplemented. See the module docstring for why this was necessary.
# =============================================================================

def run_batcs_no_predictor(E1, seed, train_set, train_eval_set, test_loader,
                            client_indices, validation_indices, model_mb,
                            budget_mb):
    method = "BATCS-NoPredictor"

    E1.set_seed(seed)

    client_loaders = [
        E1.DataLoader(
            E1.Subset(train_set, indices.tolist()),
            batch_size=E1.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        for indices in client_indices
    ]

    server_val_loader = E1.DataLoader(
        E1.Subset(train_eval_set, validation_indices.tolist()),
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    global_model = E1.CIFAR10CNN().to(E1.DEVICE)
    global_state = E1.clone_state_to_cpu(global_model.state_dict())
    worker_model = E1.CIFAR10CNN().to(E1.DEVICE)
    validation_model = E1.CIFAR10CNN().to(E1.DEVICE)

    ucb = E1.UCB1(E1.NUM_CLIENTS, seed)
    predictor = UCBOnlyProxyPredictor(mode=NOPREDICTOR_MATURITY_MODE)
    ledger = E1.CommunicationLedger(budget_mb)

    round_records: List[Dict] = []
    predictor_records: List[Dict] = []
    previous_validation_accuracy = None

    run_start = time.perf_counter()
    initial_test_accuracy = E1.evaluate_accuracy(global_model, test_loader, E1.DEVICE)

    for round_number in range(1, E1.NUM_ROUNDS + 1):
        round_start = time.perf_counter()

        # ---------------- Stage 1: identical UCB mechanism -------------------
        selected = ucb.select(E1.CLIENTS_PER_ROUND)
        if len(selected) != E1.CLIENTS_PER_ROUND:
            raise RuntimeError("Stage-1 selected client count changed.")

        downlink_mb = E1.CLIENTS_PER_ROUND * model_mb
        if not ledger.can_charge(downlink_mb):
            print(f"{method:16s} seed={seed} budget exhausted before "
                  f"round {round_number}.")
            break

        # ---------------- pre-round validation (identical to BATCS) ----------
        validation_model.load_state_dict(global_state, strict=True)
        validation_before = E1.evaluate_accuracy(validation_model, server_val_loader, E1.DEVICE)
        if previous_validation_accuracy is None:
            validation_trend_pp = 0.0
        else:
            validation_trend_pp = validation_before - previous_validation_accuracy

        # ---------------- local training (identical) --------------------------
        local_states, local_losses, first_losses, client_sizes = [], [], [], []
        for client_id in selected:
            state, avg_loss, first_loss = E1.local_train(
                worker_model, global_state, client_loaders[client_id])
            local_states.append(state)
            local_losses.append(avg_loss)
            first_losses.append(first_loss)
            client_sizes.append(len(client_indices[client_id]))

        full_sizes_mb = np.asarray(
            [E1.serialized_size_mb(state) for state in local_states],
            dtype=np.float64)

        # ---------------- preview (identical) ----------------------------------
        previews = [
            E1.compute_preview_stats(global_state, local_state, avg_loss)
            for local_state, avg_loss in zip(local_states, local_losses)
        ]
        preview_sizes_mb = np.asarray(
            [E1.preview_size_mb(p) for p in previews], dtype=np.float64)
        preview_mb = float(preview_sizes_mb.sum())

        remaining_after_mandatory = ledger.remaining() - downlink_mb - preview_mb
        if remaining_after_mandatory < 0.0:
            print(f"{method} cannot afford complete preview transmission "
                  f"at round {round_number}.")
            break

        future_rounds = E1.NUM_ROUNDS - round_number
        mean_preview_mb = float(np.mean(preview_sizes_mb))
        future_mandatory_reserve = future_rounds * (
            downlink_mb + mean_preview_mb * E1.CLIENTS_PER_ROUND)
        acquisition_budget_mb = max(0.0, remaining_after_mandatory - future_mandatory_reserve)

        features = [
            E1.preview_to_features(preview, full_sizes_mb[position])
            for position, preview in enumerate(previews)
        ]

        ucb_snapshot = ucb.scores_for(selected)

        # ---------------- Stage 2: E1's unmodified choose_batcs_stage2 -------
        # ONLY the predictor object differs from real BATCS.
        predictor.set_round_scores(ucb_snapshot)
        decision = E1.choose_batcs_stage2(
            features=features,
            update_sizes_mb=full_sizes_mb,
            ucb_scores=ucb_snapshot,
            predictor=predictor,
            round_number=round_number,
            validation_trend_pp=validation_trend_pp,
            acquisition_budget_mb=acquisition_budget_mb,
            rng=np.random.default_rng(seed + round_number + 30_000),
        )
        accepted_positions = list(decision.accepted_positions)
        stage2_mode = decision.mode

        # ---------------- communication feasibility (identical recovery) ----
        accepted_mask = np.zeros(E1.CLIENTS_PER_ROUND, dtype=bool)
        accepted_mask[accepted_positions] = True
        accepted_full_mb = float(full_sizes_mb[accepted_mask].sum())
        round_total = downlink_mb + preview_mb + accepted_full_mb

        if not ledger.can_charge(round_total):
            ranked_accepted = sorted(
                accepted_positions,
                key=lambda position: (
                    decision.utility_per_mb[position],
                    decision.combined_scores[position],
                ),
            )
            recovered = False
            while ranked_accepted:
                candidate_mask = np.zeros(E1.CLIENTS_PER_ROUND, dtype=bool)
                candidate_mask[ranked_accepted] = True
                candidate_full_mb = float(full_sizes_mb[candidate_mask].sum())
                candidate_total = downlink_mb + preview_mb + candidate_full_mb
                if ledger.can_charge(candidate_total):
                    accepted_mask = candidate_mask
                    accepted_positions = ranked_accepted.copy()
                    round_total = candidate_total
                    recovered = True
                    break
                ranked_accepted.pop()
            if not recovered:
                print(f"{method} has no affordable full update for "
                      f"round {round_number}.")
                break

        # ---------------- communication charge (identical) ----------------------
        row = ledger.charge(
            round_number=round_number,
            selected_count=len(selected),
            accepted_mask=accepted_mask,
            downlink_mb=downlink_mb,
            preview_mb=preview_mb,
            update_sizes_mb=full_sizes_mb,
            method=method,
            test_accuracy=np.nan,
            validation_before=validation_before,
            validation_trend_pp=validation_trend_pp,
            stage2_mode=stage2_mode,
        )

        pre_aggregation_global_state = {k: v.clone() for k, v in global_state.items()}

        # ---------------- aggregation (identical) --------------------------------
        if accepted_positions:
            accepted_states = [local_states[p] for p in accepted_positions]
            accepted_weights = [client_sizes[p] for p in accepted_positions]
            global_state = E1.fedavg_aggregate(accepted_states, accepted_weights)
            global_model.load_state_dict(global_state, strict=True)
        else:
            global_model.load_state_dict(global_state, strict=True)

        # ---------------- post-aggregation feedback (identical structure) -------
        accepted_set = set(accepted_positions)
        aggregate_utilities = E1.aggregate_validation_utility(
            global_state=pre_aggregation_global_state,
            local_states=local_states,
            accepted_positions=accepted_positions,
            client_weights=client_sizes,
            validation_loader=server_val_loader,
            validation_model=validation_model,
        )

        for position in range(E1.CLIENTS_PER_ROUND):
            accepted_flag = position in accepted_set

            observed_marginal_utility = (
                aggregate_utilities[position] if accepted_flag else np.nan)

            client_comm_mb = (
                model_mb
                + float(preview_sizes_mb[position])
                + float(full_sizes_mb[position])
            )

            if accepted_flag:
                reward = observed_marginal_utility / max(client_comm_mb, 1e-12)
                ucb.update(selected[position], reward)

            if method == "BATCS-NoPredictor":
                # No learned-predictor diagnostics are emitted. The ablation
                # still receives the same post-decision aggregate-marginal
                # UCB feedback as the Stage-2 controller.
                continue

            predictor_records.append({
                "Method": "BATCS",
                "Seed": seed,
                "Round": round_number,
                "Client": selected[position],
                "Predicted_Utility_pp": float(decision.predicted_utilities[position]),
                "Uncertainty_pp": float(decision.uncertainties[position]),
                "Risk_Utility_pp": float(decision.risk_utilities[position]),
                "Utility_per_MB": float(decision.utility_per_mb[position]),
                "Combined_Score": float(decision.combined_scores[position]),
                "Percentile_Threshold": float(decision.percentile_threshold),
                "Percentile_Eligible": bool(
                    decision.utility_per_mb[position] >= decision.percentile_threshold),
                "Accepted": bool(accepted_flag),
                "Observed_Marginal_Utility_pp": (
                    float(observed_marginal_utility) if accepted_flag else np.nan),
            })

        for position in accepted_positions:
            predictor.update(features[position], aggregate_utilities[position])

        validation_model.load_state_dict(global_state, strict=True)
        validation_after = E1.evaluate_accuracy(validation_model, server_val_loader, E1.DEVICE)
        previous_validation_accuracy = validation_after

        # ---------------- test accuracy (identical) -------------------------------
        test_accuracy = E1.evaluate_accuracy(global_model, test_loader, E1.DEVICE)
        row["Test_Accuracy_pct"] = test_accuracy

        row["Predictor_Samples"] = np.nan  # E1-schema-parity placeholder (unused here)
        row["Maturity_Counter_NOT_Learning"] = predictor.n_samples
        row["Learned_Ridge_Samples"] = predictor.learned_ridge_samples
        row["Learned_Ridge_Coefficients_Fitted"] = False
        row["Predictor_Confidence"] = decision.predictor_confidence
        row["Stage2_Percentile_Threshold"] = decision.percentile_threshold
        row["Stage2_Dynamic_Pool_Count"] = decision.dynamic_pool_count
        row["Stage2_Desired_Count"] = decision.desired_count
        row["Stage2_Score_Separation"] = decision.score_separation
        row["Stage2_Affordable_Count"] = decision.affordable_count
        row["Stage2_Percentile_Eligible_Count"] = decision.percentile_eligible_count
        row["Budget_Limit_MB"] = budget_mb if math.isfinite(ledger.budget_mb) else np.inf
        row["Initial_Test_Accuracy_pct"] = initial_test_accuracy
        row["Round_Time_sec"] = time.perf_counter() - round_start

        round_records.append(dict(row))

        print(f"{method:16s} seed={seed} round={round_number:02d}/{E1.NUM_ROUNDS} "
              f"selected={len(selected):2d} accepted={len(accepted_positions):2d} "
              f"acc={test_accuracy:6.2f}% comm={ledger.used_mb:8.2f} MB "
              f"[stage2={stage2_mode}]")

        if (math.isfinite(ledger.budget_mb) and ledger.remaining() < model_mb):
            print(f"    {method}: budget cannot fund another complete "
                  f"Stage-1 downlink.")
            break

    if round_records:
        final_accuracy = float(round_records[-1]["Test_Accuracy_pct"])
    else:
        final_accuracy = float(initial_test_accuracy)

    rounds_completed = len(round_records)
    total_mb = float(ledger.used_mb)
    communication_reduction = 100.0 * (
        1.0 - total_mb / max(E1.fedavg_envelope_mb_global, 1e-12))
    elapsed = time.perf_counter() - run_start

    summary = {
        "method": method,
        "seed": seed,
        "final_accuracy": final_accuracy,
        "rounds_completed": rounds_completed,
        "total_mb": total_mb,
        "communication_reduction_pct": communication_reduction,
        "accuracy_per_mb": final_accuracy / max(total_mb, 1e-12),
        "elapsed_seconds": elapsed,
        "budget_mb": budget_mb if math.isfinite(budget_mb) else np.inf,
    }

    return summary, round_records, predictor_records


# =============================================================================
# INVARIANT CHECKS (required before a job may be marked valid)
# =============================================================================

def check_invariants(method: str, seed: int, summary: Dict, round_records: List[Dict],
                      predictor_records: List[Dict], budget_mb: float) -> List[str]:
    """Returns a list of violation strings; empty list == all invariants hold."""
    problems: List[str] = []

    for row in round_records:
        # 1. exactly 10 Stage-1 clients every completed round
        if row.get("Selected") != 10:
            problems.append(f"round {row.get('Round')}: Selected != 10")
        # 2. accepted <= 10
        if not (0 <= row.get("Accepted", -1) <= 10):
            problems.append(f"round {row.get('Round')}: Accepted out of [0,10]")
        # 3. accepted count respects BATCS bounds where applicable
        if method in ("BATCS", "BATCS-NoPredictor"):
            acc = row.get("Accepted")
            mode = row.get("Stage2_Mode", "")
            # E1's own function can legitimately return fewer than
            # BATCS_MIN_ACCEPT under "budget_limited"/"no_dynamic_capacity"/
            # "budget_no_capacity"/"percentile_no_candidate" -- those are
            # hard-budget/no-candidate floors, not a violation of the 4-7
            # *evidence* window. Only flag an out-of-window count when the
            # mode indicates the evidence-driven path actually ran.
            evidence_modes = {
                "evidence_adaptive", "strong_evidence_selective",
                "validation_conservative", "uncertainty_conservative",
                "weak_score_separation", "predictor_low_confidence",
            }
            if mode in evidence_modes and not (4 <= acc <= 7):
                problems.append(
                    f"round {row.get('Round')}: accepted={acc} outside [4,7] "
                    f"for evidence mode '{mode}'")
        # 4. cumulative communication <= B0
        cum = row.get("Cumulative_Communication_MB", 0.0)
        if math.isfinite(budget_mb) and cum > budget_mb + 1e-6:
            problems.append(f"round {row.get('Round')}: cumulative {cum} > B0 {budget_mb}")
        # 5. per-round communication identity holds
        total = row.get("Round_Communication_MB", None)
        expected = (row.get("Downlink_MB", 0.0) + row.get("Preview_MB", 0.0)
                    + row.get("Accepted_Full_MB", 0.0))
        if total is None or abs(total - expected) > 1e-6:
            problems.append(f"round {row.get('Round')}: communication identity violated")

    # 6. no NaN/Inf final accuracy
    fa = summary.get("final_accuracy")
    if fa is None or not math.isfinite(fa):
        problems.append("final_accuracy is NaN/Inf")

    # 7. rounds_completed <= 50
    if summary.get("rounds_completed", 0) > 50:
        problems.append("rounds_completed > 50")

    # 8. method and seed metadata match requested job
    if summary.get("method") != method:
        problems.append(f"summary method '{summary.get('method')}' != requested '{method}'")
    if summary.get("seed") != seed:
        problems.append(f"summary seed {summary.get('seed')} != requested {seed}")

    # 9. predictor sample count (see module docstring's flagged conflict):
    #    literal reading only applies under NOPREDICTOR_MATURITY_MODE="pinned_zero".
    if method == "BATCS-NoPredictor":
        for row in round_records:
            if row.get("Learned_Ridge_Samples", 0) != 0:
                problems.append("BATCS-NoPredictor: Learned_Ridge_Samples != 0 "
                                 "(a learned model was fit -- should never happen)")
            if row.get("Learned_Ridge_Coefficients_Fitted", True) is not False:
                problems.append("BATCS-NoPredictor: Learned_Ridge_Coefficients_Fitted != False")
        if NOPREDICTOR_MATURITY_MODE == "pinned_zero":
            for row in round_records:
                if row.get("Maturity_Counter_NOT_Learning", -1) != 0:
                    problems.append(
                        "BATCS-NoPredictor (pinned_zero mode): maturity counter "
                        "advanced but was required to stay at 0")

    # 10. BATCS predictor targets are post-decision only -- structural
    #     invariant enforced by construction (predictor.update() is only
    #     ever called, in both E1.run_method's BATCS branch and this
    #     file's run_batcs_no_predictor, AFTER aggregate_validation_utility()
    #     -- which itself requires accepted_positions to already be fixed).
    #     Verified here only as a schema sanity check:
    if method in ("BATCS", "BATCS-NoPredictor"):
        pr_df_cols = set()
        for rec in predictor_records:
            pr_df_cols.update(rec.keys())
        required_cols = {"Accepted", "Observed_Marginal_Utility_pp", "Isolated_Utility_pp"}
        if predictor_records and not required_cols.issubset(pr_df_cols):
            problems.append("predictor_records missing required post-decision columns")

    # 11. final accuracy comes from final global model evaluation
    if round_records:
        if abs(round_records[-1].get("Test_Accuracy_pct", float("nan"))
               - summary.get("final_accuracy", float("nan"))) > 1e-9:
            problems.append("final_accuracy does not match last round's Test_Accuracy_pct")

    return problems


# =============================================================================
# SINGLE-JOB WORKER (runs inside its own subprocess)
# =============================================================================

def run_single_job(method: str, seed: int, out_path: Path) -> None:
    """Executes exactly one method x seed job, writes an atomic result file
    on success, and lets any exception propagate (-> non-zero exit code
    for the launcher to detect)."""
    E1, e1_hash = load_e1_module()

    required = [
        "set_seed", "load_cifar10", "create_validation_split",
        "dirichlet_partition", "CIFAR10CNN", "clone_state_to_cpu",
        "serialized_size_mb", "calculate_budget", "DataLoader", "Subset",
        "UCB1", "CommunicationLedger", "local_train", "compute_preview_stats",
        "preview_size_mb", "preview_to_features", "choose_batcs_stage2",
        "fedavg_aggregate", "aggregate_validation_utility", "evaluate_accuracy",
        "NUM_CLIENTS", "CLIENTS_PER_ROUND", "NUM_ROUNDS", "BATCH_SIZE",
        "SERVER_VALIDATION_SIZE", "DIRICHLET_ALPHA", "DEVICE",
    ]
    missing = [name for name in required if not hasattr(E1, name)]
    if missing:
        raise AttributeError(
            "Frozen E1.py is missing required symbols; refusing to invent APIs: "
            + ", ".join(missing)
        )

    print(f"[worker] method={method} seed={seed} E1_sha256={e1_hash[:12]}...")
    print(f"[worker] device={E1.DEVICE}")

    train_set, train_eval_set, test_set = E1.load_cifar10()
    labels = np.asarray(train_set.targets, dtype=np.int64)
    test_loader = E1.DataLoader(test_set, batch_size=512, shuffle=False,
                                 num_workers=0, pin_memory=False)

    measurement_model = E1.CIFAR10CNN()
    measurement_state = E1.clone_state_to_cpu(measurement_model.state_dict())
    model_mb = E1.serialized_size_mb(measurement_state)

    hard_budget_mb, fedavg_envelope_mb = E1.calculate_budget(model_mb)
    # REQUIRED: E1.run_method() reads this module-level global directly;
    # it is only ever set inside E1.main(), which we deliberately never
    # call (per-job process isolation). Setting it here is gluing, not a
    # change to E1's definitions -- same function, same formula, same
    # value E1.main() would have produced.
    E1.fedavg_envelope_mb_global = fedavg_envelope_mb

    validation_indices, client_pool = E1.create_validation_split(
        labels, E1.SERVER_VALIDATION_SIZE, seed)
    client_indices = E1.dirichlet_partition(
        labels, client_pool, E1.NUM_CLIENTS, E1.DIRICHLET_ALPHA, seed)

    if method in ("UCB+Budget", "BATCS"):
        # Called completely unmodified -- these method strings are
        # natively recognized by E1.run_method()'s own dispatch.
        summary, round_records, predictor_records = E1.run_method(
            method=method, seed=seed, train_set=train_set,
            train_eval_set=train_eval_set, test_loader=test_loader,
            client_indices=client_indices, validation_indices=validation_indices,
            model_mb=model_mb, budget_mb=hard_budget_mb)
    elif method == "BATCS-NoPredictor":
        summary, round_records, predictor_records = run_batcs_no_predictor(
            E1, seed, train_set, train_eval_set, test_loader,
            client_indices, validation_indices, model_mb, hard_budget_mb)
    else:
        raise ValueError(f"Unknown E2 method: {method}")

    problems = check_invariants(method, seed, summary, round_records,
                                 predictor_records, hard_budget_mb)
    if problems:
        raise RuntimeError(
            f"Invariant check failed for {method}/{seed}: " + "; ".join(problems))

    payload = {
        "method": method,
        "seed": seed,
        "summary": summary,
        "round_records": round_records,
        "predictor_records": predictor_records,
        "invariants_passed": True,
        "nopredictor_maturity_mode": (
            NOPREDICTOR_MATURITY_MODE if method == "BATCS-NoPredictor" else None),
    }

    # Atomic write: write to a temp file in the same directory, then
    # os.replace (atomic on POSIX and Windows within the same filesystem).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    os.replace(tmp_path, out_path)

    print(f"[worker] SUCCESS method={method} seed={seed} "
          f"acc={summary['final_accuracy']:.2f}% total_mb={summary['total_mb']:.2f} "
          f"-> {out_path}")

    # Best-effort cleanup before process exit; not a segfault guarantee.
    del train_set, train_eval_set, test_set, test_loader
    gc.collect()
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
            except Exception:
                pass
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    except Exception:
        pass


def _json_default(o):
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not math.isfinite(v) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(f"Not JSON serializable: {type(o)}")


def _job_result_path(method: str, seed: int) -> Path:
    safe_method = method.replace("+", "plus").replace(" ", "_")
    return JOB_DIR / f"{safe_method}__seed{seed}.json"


def _validate_existing_job_file(path: Path, method: str, seed: int) -> bool:
    """Resume support: a previously completed job file is reused only if
    it parses, matches the requested method/seed, and re-passes invariant
    checks -- never trusted blindly."""
    if not path.is_file():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("method") != method or payload.get("seed") != seed:
            return False
        if not payload.get("invariants_passed"):
            return False
        problems = check_invariants(
            method, seed, payload["summary"], payload["round_records"],
            payload["predictor_records"],
            payload["summary"].get("budget_mb", float("inf")))
        return not problems
    except Exception:
        return False


# =============================================================================
# LAUNCHER: subprocess isolation, retries, resume, aggregation
# =============================================================================

def launch_job(method: str, seed: int) -> bool:
    """Runs one method x seed job in up to (1 + MAX_RETRIES) fresh
    subprocesses. Returns True iff a valid result file exists at the end."""
    out_path = _job_result_path(method, seed)

    if _validate_existing_job_file(out_path, method, seed):
        print(f"[launcher] REUSING valid prior result: {method}/{seed}")
        return True

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe_method = method.replace("+", "plus").replace(" ", "_")

    for attempt in range(1, MAX_RETRIES + 2):  # initial try + MAX_RETRIES retries
        log_path = LOG_DIR / f"{safe_method}__seed{seed}__attempt{attempt}.log"
        print(f"[launcher] START {method}/{seed} attempt {attempt} "
              f"(log: {log_path})")

        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--worker", "--method", method, "--seed", str(seed),
               "--out", str(out_path)]

        env = dict(os.environ)
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["MPLBACKEND"] = "Agg"

        with open(log_path, "w", encoding="utf-8") as log_file:
            try:
                proc = subprocess.run(
                    cmd, stdout=log_file, stderr=subprocess.STDOUT,
                    env=env, timeout=None,
                )
                returncode = proc.returncode
            except Exception as exc:
                log_file.write(f"\n[launcher] subprocess.run raised: {exc}\n")
                traceback.print_exc(file=log_file)
                returncode = -1

        crashed = returncode < 0  # negative => killed by signal (e.g. segfault)
        failed = returncode != 0

        if not failed and _validate_existing_job_file(out_path, method, seed):
            print(f"[launcher] SUCCESS {method}/{seed} (attempt {attempt}, "
                  f"exit={returncode})")
            return True

        note = "SEGFAULT/SIGNAL" if crashed else f"exit={returncode}"
        print(f"[launcher] FAILED {method}/{seed} attempt {attempt} ({note}); "
              f"see {log_path}")

        # Never accept a partial/invalid result file left behind by a
        # crashed attempt.
        if out_path.is_file() and not _validate_existing_job_file(out_path, method, seed):
            try:
                out_path.unlink()
            except OSError:
                pass

    print(f"[launcher] GIVING UP on {method}/{seed} after "
          f"{MAX_RETRIES + 1} attempts. Marked INVALID.")
    return False


def build_seed_table(all_summaries: List[Dict]) -> pd.DataFrame:
    rows = []
    for method in E2_METHODS:
        subset = [s for s in all_summaries if s["method"] == method]
        if not subset:
            continue
        acc = np.asarray([s["final_accuracy"] for s in subset], dtype=np.float64)
        mb = np.asarray([s["total_mb"] for s in subset], dtype=np.float64)
        red = np.asarray([s["communication_reduction_pct"] for s in subset], dtype=np.float64)
        acc_mean, acc_std = acc.mean(), (acc.std(ddof=1) if len(acc) > 1 else 0.0)
        mb_mean, mb_std = mb.mean(), (mb.std(ddof=1) if len(mb) > 1 else 0.0)
        rows.append({
            "Method": method,
            "Accuracy": f"{acc_mean:.2f} ± {acc_std:.2f}",
            "Total_MB": f"{mb_mean:.2f} ± {mb_std:.2f}",
            "Communication_Reduction_pct": f"{red.mean():.2f}",
            "Accuracy_per_MB": f"{(acc_mean / max(mb_mean, 1e-12)):.4f}",
        })
    return pd.DataFrame(rows)


def main_launcher() -> int:
    print("=" * 100)
    print("BATCS -- E2: STAGE-2 NOVELTY ISOLATION")
    print("=" * 100)

    if not E1_PATH.is_file():
        print(f"ERROR: frozen E1.py not found at {E1_PATH}", file=sys.stderr)
        return 2

    # IMPORTANT: the parent launcher does NOT import E1 / torch / MPS.
    # Every method×seed worker gets a completely fresh Python + MPS process.
    e1_hash = hashlib.sha256(E1_PATH.read_bytes()).hexdigest()
    print(f"Frozen E1.py: {E1_PATH}  sha256={e1_hash}")
    print(f"E2 methods: {E2_METHODS}")
    print(f"E2 seeds: {E2_SEEDS}")
    print(f"NoPredictor maturity mode: {NOPREDICTOR_MATURITY_MODE}")

    jobs = [(m, s) for m in E2_METHODS for s in E2_SEEDS]
    print(f"Total jobs: {len(jobs)}")

    job_status: Dict[Tuple[str, int], bool] = {}
    for method, seed in jobs:
        job_status[(method, seed)] = launch_job(method, seed)

    invalid_jobs = [k for k, ok in job_status.items() if not ok]
    if invalid_jobs:
        print("\n" + "=" * 100)
        print("E2 ABORTED: the following jobs never produced a valid result "
              "after retries. NOTHING is aggregated -- per spec, no failed "
              "subprocess result may enter the final summary.")
        for method, seed in invalid_jobs:
            print(f"    INVALID: {method} / seed={seed}")
        print("=" * 100)
        _write_manifest(e1_hash, job_status, aggregated=False)
        return 1

    # ---------------- aggregate only after all 9 jobs are valid --------------
    all_summaries: List[Dict] = []
    all_round_records: List[Dict] = []
    all_predictor_records: List[Dict] = []

    for method, seed in jobs:
        with open(_job_result_path(method, seed), "r", encoding="utf-8") as f:
            payload = json.load(f)
        all_summaries.append(payload["summary"])
        all_round_records.extend(payload["round_records"])
        all_predictor_records.extend(payload["predictor_records"])

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    seed_df = pd.DataFrame(all_summaries)
    seed_df.to_csv(RESULT_DIR / "E2_seed_results.csv", index=False)

    round_df = pd.DataFrame(all_round_records)
    round_df.to_csv(RESULT_DIR / "E2_round_results.csv", index=False)

    batcs_predictor_records = [
        r for r in all_predictor_records
        if any(s["method"] == "BATCS" and s["seed"] == r["Seed"] for s in all_summaries)
    ]
    # NOTE: predictor_records don't carry a Method field; filter using the
    # job structure instead, since BATCS and BATCS-NoPredictor jobs are
    # written to disjoint payloads and never mixed here.
    batcs_only_predictor_records = []
    for method, seed in jobs:
        if method != "BATCS":
            continue
        with open(_job_result_path(method, seed), "r", encoding="utf-8") as f:
            payload = json.load(f)
        batcs_only_predictor_records.extend(payload["predictor_records"])
    pd.DataFrame(batcs_only_predictor_records).to_csv(
        RESULT_DIR / "E2_BATCS_predictor.csv", index=False)

    table_df = build_seed_table(all_summaries)
    table_df.to_csv(RESULT_DIR / "E2_Table.csv", index=False)

    # ---------------- paired BATCS vs BATCS-NoPredictor -----------------------
    batcs_map = {s["seed"]: s["final_accuracy"] for s in all_summaries if s["method"] == "BATCS"}
    nopred_map = {s["seed"]: s["final_accuracy"] for s in all_summaries if s["method"] == "BATCS-NoPredictor"}
    batcs_mb_map = {s["seed"]: s["total_mb"] for s in all_summaries if s["method"] == "BATCS"}
    nopred_mb_map = {s["seed"]: s["total_mb"] for s in all_summaries if s["method"] == "BATCS-NoPredictor"}
    common_seeds = sorted(set(batcs_map) & set(nopred_map))

    acc_diffs = [batcs_map[s] - nopred_map[s] for s in common_seeds]
    mb_diffs = [batcs_mb_map[s] - nopred_mb_map[s] for s in common_seeds]

    comparison = {
        "method_a": "BATCS",
        "method_b": "BATCS-NoPredictor",
        "paired_seeds": common_seeds,
        "accuracy_difference_per_seed_pp": {
            str(s): (batcs_map[s] - nopred_map[s]) for s in common_seeds},
        "accuracy_difference_mean_pp": float(np.mean(acc_diffs)) if acc_diffs else None,
        "communication_difference_per_seed_mb": {
            str(s): (batcs_mb_map[s] - nopred_mb_map[s]) for s in common_seeds},
        "communication_difference_mean_mb": float(np.mean(mb_diffs)) if mb_diffs else None,
        "note": ("Descriptive comparison only; BATCS is not forced to "
                 "outperform BATCS-NoPredictor. If BATCS is worse, that is "
                 "reported honestly."),
    }
    with open(RESULT_DIR / "E2_BATCS_vs_BATCS_NoPredictor.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    _write_manifest(e1_hash, job_status, aggregated=True)

    print("\n" + "=" * 100)
    print("TABLE — E2 STAGE-2 NOVELTY ISOLATION")
    print("=" * 100)
    print(table_df.to_string(index=False))
    print("\n" + "=" * 100)
    print("BATCS vs BATCS-NoPredictor (paired)")
    print("=" * 100)
    print(json.dumps(comparison, indent=2))
    print(f"\nResults saved in: {RESULT_DIR.resolve()}")
    return 0


def _write_manifest(e1_hash: str, job_status: Dict[Tuple[str, int], bool],
                     aggregated: bool) -> None:
    manifest = {
        "e1_source_path": str(E1_PATH),
        "e1_source_sha256": e1_hash,
        "e2_code_version": "1.0.0",
        "e2_seeds": E2_SEEDS,
        "e2_methods": E2_METHODS,
        "nopredictor_maturity_mode": NOPREDICTOR_MATURITY_MODE,
        "timestamp_unix": time.time(),
        "job_status": {f"{m}/{s}": ok for (m, s), ok in job_status.items()},
        "all_jobs_valid": aggregated,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="BATCS E2 — Stage-2 novelty isolation")
    parser.add_argument("--worker", action="store_true",
                         help="Internal: run a single method/seed job (used by the launcher).")
    parser.add_argument("--method", type=str, default=None, choices=E2_METHODS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.worker:
        if args.method is None or args.seed is None or args.out is None:
            print("--worker requires --method, --seed, and --out", file=sys.stderr)
            sys.exit(2)
        try:
            run_single_job(args.method, args.seed, Path(args.out))
            sys.exit(0)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    else:
        sys.exit(main_launcher())