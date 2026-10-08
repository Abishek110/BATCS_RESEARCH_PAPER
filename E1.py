#!/usr/bin/env python3
"""
BATCS — E1 MAIN EXPERIMENT
Budget-Coupled Two-Stage Communication Control for Federated Learning
using Multi-Armed Bandits.

E1:
    Dataset: CIFAR-10
    Clients: 50
    Stage-1 selected clients: 10 (20%)
    Rounds: 50
    Local epochs: 1
    Batch size: 64
    Learning rate: 0.01
    Momentum: 0.9
    Dirichlet alpha: 0.5
    Seeds: 42, 123, 2024

Methods:
    FedAvg
    Random+Budget
    UCB
    UCB+Budget
    BATCS

Layer A — Stage 1:
    EXACTLY 10 of 50 clients are selected before local training.

Layer B — BATCS Stage 2:
    local training
        -> lightweight scalar preview
        -> learned utility prediction
        -> uncertainty-aware utility
        -> percentile evidence boundary
        -> dynamic acceptance count
        -> controlled borderline exploration
        -> exact budget-feasible subset acquisition
        -> aggregation of accepted updates only
        -> post-aggregation marginal validation feedback

Layer C — E1 experimental configuration:
    kept unchanged from the supplied code.

Communication:
    B_used =
        B_downlink
        + B_preview
        + B_accepted_full

Rejected full updates are NOT charged.

Important:
    - No test-set data is used to train the predictor.
    - Predictor targets are aggregate-marginal validation gains.
    - Predictor/UCB feedback is observed only after Stage-2 selection.
    - Current-round feedback cannot influence the current-round decision.
    - BATCS is not forced to outperform UCB+Budget.
"""

from __future__ import annotations

import io
import json
import math
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# CONFIGURATION — FROZEN E1 SETTINGS
# ============================================================

NUM_CLIENTS = 50
CLIENTS_PER_ROUND = 10
NUM_ROUNDS = 50

LOCAL_EPOCHS = 1
BATCH_SIZE = 64

LEARNING_RATE = 0.01
MOMENTUM = 0.9

DIRICHLET_ALPHA = 0.5
SERVER_VALIDATION_SIZE = 2000

SEEDS = [42, 123, 2024, 7, 2025]

MAIN_BUDGET_RATIO = 0.50

DATA_DIR = Path("./data")
RESULT_DIR = Path("./batcs_results_final")


# ============================================================
# DEVICE
# ============================================================

if (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# BATCS STAGE-2 PARAMETERS
# ============================================================

BATCS_MIN_ACCEPT = 4
BATCS_MAX_ACCEPT = 7

# Layer-B safety controls: remain within the existing 4-7 design,
# but only unlock 6-7 when evidence is strong and the predictor is mature.
BATCS_SOFT_MAX_ACCEPT = 5
BATCS_WARMUP_ROUNDS = 8
BATCS_MIN_PREDICTOR_SAMPLES = 40

# Percentile used to identify the strong-candidate region.
# It is an evidence boundary, not a hard cardinality constraint.
BATCS_PERCENTILE = 50.0

# Number of borderline candidates that may be retained
# below the percentile boundary so that the dynamic
# acceptance controller can genuinely choose 4-7 clients.
BATCS_BORDERLINE_POOL = 3

# Optimistic uncertainty treatment:
# predicted utility + beta * uncertainty.
BATCS_UNCERTAINTY_BETA = 0.20

# Maximum influence of the learned predictor relative to UCB history.
BATCS_MAX_PREDICTOR_WEIGHT = 0.35

# Controlled exploration of borderline candidates.
BATCS_EXPLORATION_PROB = 0.05

# Validation deterioration trigger.
BATCS_BAD_TREND_THRESHOLD_PP = 0.50

# Dynamic count parameters.
BATCS_LOW_SCORE_SEPARATION = 0.10
BATCS_HIGH_SCORE_SEPARATION = 0.30

# Preview protocol overhead.
PREVIEW_HEADER_BYTES = 256


# ============================================================
# DIRECTORIES
# ============================================================

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(
    seed: int,
) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(
        torch.backends,
        "cudnn",
    ):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

def load_cifar10():

    mean = (
        0.4914,
        0.4822,
        0.4465,
    )

    std = (
        0.2470,
        0.2435,
        0.2616,
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(
            32,
            padding=4,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean,
            std,
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean,
            std,
        ),
    ])

    train_set = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=train_transform,
    )

    train_eval_set = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=False,
        transform=eval_transform,
    )

    test_set = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=eval_transform,
    )

    if len(train_set) != 50_000:
        raise RuntimeError(
            f"Unexpected CIFAR-10 training size: {len(train_set)}"
        )

    if len(test_set) != 10_000:
        raise RuntimeError(
            f"Unexpected CIFAR-10 test size: {len(test_set)}"
        )

    return (
        train_set,
        train_eval_set,
        test_set,
    )


# ============================================================
# SERVER VALIDATION SPLIT
# ============================================================

def create_validation_split(
    labels: np.ndarray,
    validation_size: int,
    seed: int,
):

    if validation_size <= 0:
        raise ValueError(
            "validation_size must be positive."
        )

    if validation_size >= len(labels):
        raise ValueError(
            "Validation set must be smaller than training set."
        )

    classes = np.unique(labels)

    if validation_size % len(classes) != 0:
        raise ValueError(
            "SERVER_VALIDATION_SIZE must be divisible by number of classes."
        )

    rng = np.random.default_rng(seed)

    per_class = (
        validation_size
        //
        len(classes)
    )

    validation_indices = []

    for cls in classes:

        cls_indices = np.where(
            labels == cls
        )[0].copy()

        rng.shuffle(
            cls_indices
        )

        validation_indices.extend(
            cls_indices[
                :per_class
            ].tolist()
        )

    validation_indices = np.asarray(
        validation_indices,
        dtype=np.int64,
    )

    all_indices = np.arange(
        len(labels),
        dtype=np.int64,
    )

    mask = np.ones(
        len(labels),
        dtype=bool,
    )

    mask[
        validation_indices
    ] = False

    client_pool = all_indices[
        mask
    ]

    rng.shuffle(
        validation_indices
    )

    rng.shuffle(
        client_pool
    )

    if len(validation_indices) != validation_size:
        raise RuntimeError(
            "Validation split size mismatch."
        )

    if (
        np.intersect1d(
            validation_indices,
            client_pool,
        ).size
        != 0
    ):
        raise RuntimeError(
            "Validation and client-pool overlap detected."
        )

    return (
        validation_indices,
        client_pool,
    )


# ============================================================
# DIRICHLET PARTITION
# ============================================================

def dirichlet_partition(
    labels: np.ndarray,
    pool_indices: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
):

    if num_clients <= 0:
        raise ValueError(
            "num_clients must be positive."
        )

    if alpha <= 0:
        raise ValueError(
            "alpha must be positive."
        )

    rng = np.random.default_rng(seed)

    clients: List[List[int]] = [
        []
        for _ in range(num_clients)
    ]

    pool_labels = labels[
        pool_indices
    ]

    for cls in np.unique(
        pool_labels
    ):

        cls_indices = pool_indices[
            pool_labels == cls
        ].copy()

        rng.shuffle(
            cls_indices
        )

        proportions = rng.dirichlet(
            np.ones(num_clients)
            *
            alpha
        )

        split_points = (
            np.cumsum(
                proportions
            )
            *
            len(cls_indices)
        ).astype(int)

        split_points = np.clip(
            split_points,
            0,
            len(cls_indices),
        )

        chunks = np.split(
            cls_indices,
            split_points[:-1],
        )

        for client_id, chunk in enumerate(
            chunks
        ):

            clients[
                client_id
            ].extend(
                chunk.tolist()
            )

    # Repair empty client if necessary.
    for client_id in range(
        num_clients
    ):

        if clients[
            client_id
        ]:
            continue

        donor = int(
            np.argmax(
                [
                    len(x)
                    for x in clients
                ]
            )
        )

        if len(
            clients[donor]
        ) <= 1:
            raise RuntimeError(
                "Unable to repair empty client shard."
            )

        clients[
            client_id
        ].append(
            clients[
                donor
            ].pop()
        )

    for client_id in range(
        num_clients
    ):

        rng.shuffle(
            clients[
                client_id
            ]
        )

    arrays = [
        np.asarray(
            client,
            dtype=np.int64,
        )
        for client in clients
    ]

    merged = np.concatenate(
        arrays
    )

    if len(merged) != len(pool_indices):
        raise RuntimeError(
            "Partition does not cover the client pool."
        )

    if len(
        np.unique(merged)
    ) != len(merged):
        raise RuntimeError(
            "Duplicate client indices detected."
        )

    if set(
        merged.tolist()
    ) != set(
        pool_indices.tolist()
    ):
        raise RuntimeError(
            "Partition coverage mismatch."
        )

    return arrays


# ============================================================
# MODEL
# ============================================================

class CIFAR10CNN(nn.Module):

    def __init__(
        self,
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                3,
                padding=1,
            ),

            nn.ReLU(
                inplace=False
            ),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1,
            ),

            nn.ReLU(
                inplace=False
            ),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                64,
                3,
                padding=1,
            ),

            nn.ReLU(
                inplace=False
            ),

            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 4 * 4,
                128,
            ),

            nn.ReLU(
                inplace=False
            ),

            nn.Linear(
                128,
                10,
            ),
        )

    def forward(
        self,
        x,
    ):

        x = self.features(x)

        return self.classifier(x)


# ============================================================
# STATE / SERIALIZATION
# ============================================================

def clone_state_to_cpu(
    state,
):

    return {
        key:
            value.detach().cpu().clone()
        for key, value
        in state.items()
    }


def serialize_state(
    state,
):

    buffer = io.BytesIO()

    torch.save(
        state,
        buffer,
    )

    return buffer.getvalue()


def serialized_size_bytes(
    state,
):

    return len(
        serialize_state(
            state
        )
    )


def serialized_size_mb(
    state,
):

    return (
        serialized_size_bytes(
            state
        )
        /
        (1024 ** 2)
    )


# ============================================================
# FEDAVG
# ============================================================

def fedavg_aggregate(
    client_states: Sequence[
        Dict[str, torch.Tensor]
    ],
    weights: Sequence[int],
):

    if not client_states:
        raise ValueError(
            "FedAvg received no client states."
        )

    if len(client_states) != len(weights):
        raise ValueError(
            "FedAvg state/weight mismatch."
        )

    total_weight = float(
        sum(weights)
    )

    if total_weight <= 0:
        raise ValueError(
            "FedAvg total weight must be positive."
        )

    result = {}

    reference = client_states[0]

    for key in reference:

        reference_tensor = reference[
            key
        ]

        if torch.is_floating_point(
            reference_tensor
        ):

            aggregate = torch.zeros_like(
                reference_tensor,
                dtype=torch.float32,
            )

            for state, weight in zip(
                client_states,
                weights,
            ):

                aggregate += (
                    state[key].float()
                    *
                    (
                        float(weight)
                        /
                        total_weight
                    )
                )

            result[key] = aggregate.to(
                reference_tensor.dtype
            )

        else:

            largest = int(
                np.argmax(weights)
            )

            result[key] = (
                client_states[
                    largest
                ][key].clone()
            )

    return result


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_accuracy(
    model,
    loader,
    device,
):

    model.eval()

    correct = 0
    total = 0

    for x, y in loader:

        x = x.to(
            device,
            non_blocking=(
                device.type == "cuda"
            ),
        )

        y = y.to(
            device,
            non_blocking=(
                device.type == "cuda"
            ),
        )

        output = model(x)

        prediction = output.argmax(
            dim=1
        )

        correct += int(
            (
                prediction
                ==
                y
            ).sum().item()
        )

        total += int(
            y.size(0)
        )

    return (
        100.0
        *
        correct
        /
        max(
            total,
            1,
        )
    )


# ============================================================
# LOCAL TRAINING
# ============================================================

def local_train(
    worker,
    global_state,
    loader,
):

    worker.load_state_dict(
        global_state,
        strict=True,
    )

    worker.train()

    optimizer = optim.SGD(
        worker.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
    )

    total_loss = 0.0
    num_batches = 0
    first_loss = None

    for _ in range(
        LOCAL_EPOCHS
    ):

        for x, y in loader:

            x = x.to(
                DEVICE,
                non_blocking=(
                    DEVICE.type == "cuda"
                ),
            )

            y = y.to(
                DEVICE,
                non_blocking=(
                    DEVICE.type == "cuda"
                ),
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = worker(x)

            loss = F.cross_entropy(
                output,
                y,
            )

            loss.backward()

            optimizer.step()

            value = float(
                loss.detach().item()
            )

            if first_loss is None:
                first_loss = value

            total_loss += value
            num_batches += 1

    average_loss = (
        total_loss
        /
        max(
            num_batches,
            1,
        )
    )

    if first_loss is None:
        first_loss = average_loss

    return (
        clone_state_to_cpu(
            worker.state_dict()
        ),
        float(average_loss),
        float(first_loss),
    )


# ============================================================
# PREVIEW
# ============================================================

def compute_preview_stats(
    global_state,
    local_state,
    local_loss,
):

    squared_sum = 0.0
    absolute_sum = 0.0
    total_values = 0
    near_zero = 0

    for key in global_state:

        if not torch.is_floating_point(
            global_state[key]
        ):
            continue

        delta = (
            local_state[key].float()
            -
            global_state[key].float()
        )

        flat = delta.reshape(-1)

        squared_sum += float(
            torch.sum(
                flat * flat
            ).item()
        )

        absolute_sum += float(
            torch.sum(
                torch.abs(flat)
            ).item()
        )

        total_values += flat.numel()

        near_zero += int(
            torch.sum(
                torch.abs(flat)
                <
                1e-6
            ).item()
        )

    return {
        "l2_norm":
            math.sqrt(
                max(
                    squared_sum,
                    0.0,
                )
            ),

        "l1_norm":
            absolute_sum,

        "sparsity":
            (
                near_zero
                /
                max(
                    total_values,
                    1,
                )
            ),

        "local_loss":
            float(local_loss),
    }


def serialize_preview(
    preview,
):

    payload = {
        "l2_norm":
            float(
                preview["l2_norm"]
            ),

        "l1_norm":
            float(
                preview["l1_norm"]
            ),

        "sparsity":
            float(
                preview["sparsity"]
            ),

        "local_loss":
            float(
                preview["local_loss"]
            ),
    }

    return (
        pickle.dumps(
            payload,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        +
        bytes(
            PREVIEW_HEADER_BYTES
        )
    )


def preview_size_mb(
    preview,
):

    return (
        len(
            serialize_preview(
                preview
            )
        )
        /
        (1024 ** 2)
    )


def preview_to_features(
    preview,
    update_mb,
):

    return np.asarray(
        [
            np.log1p(
                max(
                    preview["l2_norm"],
                    0.0,
                )
            ),

            np.log1p(
                max(
                    preview["l1_norm"],
                    0.0,
                )
            ),

            float(
                preview["sparsity"]
            ),

            np.log1p(
                max(
                    preview["local_loss"],
                    0.0,
                )
            ),

            np.log1p(
                max(
                    update_mb,
                    1e-12,
                )
            ),
        ],
        dtype=np.float64,
    )


# ============================================================
# ONLINE RIDGE PREDICTOR
# ============================================================

class OnlineRidgePredictor:

    def __init__(
        self,
        num_features=5,
        regularization=1.0,
    ):

        self.d = (
            num_features
            +
            1
        )

        self.regularization = float(
            regularization
        )

        self.A = (
            self.regularization
            *
            np.eye(
                self.d,
                dtype=np.float64,
            )
        )

        self.b = np.zeros(
            self.d,
            dtype=np.float64,
        )

        self.n_samples = 0

        self.sse = 0.0

    def _augment(
        self,
        x,
    ):

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        if x.shape != (
            self.d - 1,
        ):

            raise ValueError(
                "Predictor feature dimension mismatch."
            )

        return np.concatenate(
            [
                x,
                np.asarray(
                    [1.0],
                    dtype=np.float64,
                ),
            ]
        )

    def coefficients(
        self,
    ):

        return np.linalg.solve(
            self.A,
            self.b,
        )

    def predict_with_uncertainty(
        self,
        x,
    ):

        z = self._augment(x)

        theta = self.coefficients()

        prediction = float(
            z
            @
            theta
        )

        leverage = float(
            z
            @
            np.linalg.solve(
                self.A,
                z,
            )
        )

        degrees = max(
            self.n_samples
            -
            self.d,
            1,
        )

        residual_scale = math.sqrt(
            max(
                self.sse
                /
                degrees,
                1e-4,
            )
        )

        uncertainty = (
            residual_scale
            *
            math.sqrt(
                max(
                    leverage,
                    0.0,
                )
            )
        )

        uncertainty *= (
            1.0
            +
            1.0
            /
            math.sqrt(
                max(
                    self.n_samples,
                    1,
                )
            )
        )

        return (
            prediction,
            float(uncertainty),
        )

    def update(
        self,
        x,
        target,
    ):

        z = self._augment(x)

        theta = self.coefficients()

        old_prediction = float(
            z
            @
            theta
        )

        residual = (
            float(target)
            -
            old_prediction
        )

        self.A += np.outer(
            z,
            z,
        )

        self.b += (
            z
            *
            float(target)
        )

        self.sse += (
            residual
            *
            residual
        )

        self.n_samples += 1

        return residual


# ============================================================
# UCB1
# ============================================================

class UCB1:

    def __init__(
        self,
        num_clients,
        seed,
    ):

        self.num_clients = int(
            num_clients
        )

        self.counts = np.zeros(
            num_clients,
            dtype=np.int64,
        )

        self.reward_sums = np.zeros(
            num_clients,
            dtype=np.float64,
        )

        self.t = 0

        self.rng = np.random.default_rng(
            seed
        )

    def _scores(
        self,
    ):

        observed = (
            self.counts
            >
            0
        )

        means = np.divide(
            self.reward_sums,
            self.counts,
            out=np.zeros_like(
                self.reward_sums
            ),
            where=observed,
        )

        bonus = np.full(
            self.num_clients,
            np.inf,
            dtype=np.float64,
        )

        if self.t > 0:

            bonus[
                observed
            ] = np.sqrt(
                2.0
                *
                np.log(
                    self.t
                    +
                    1.0
                )
                /
                np.maximum(
                    self.counts[
                        observed
                    ],
                    1,
                )
            )

        return (
            means
            +
            bonus
        )

    def select(
        self,
        k,
    ):

        self.t += 1

        scores = self._scores()

        finite_scores = np.where(
            np.isfinite(scores),
            scores,
            np.finfo(
                np.float64
            ).max,
        )

        jitter = (
            self.rng.random(
                self.num_clients
            )
            *
            1e-12
        )

        order = np.argsort(
            -(
                finite_scores
                +
                jitter
            ),
            kind="mergesort",
        )

        return (
            order[
                :k
            ]
            .astype(int)
            .tolist()
        )

    def scores_for(
        self,
        client_ids,
    ):

        scores = self._scores()

        ids = np.asarray(
            client_ids,
            dtype=np.int64,
        )

        values = scores[
            ids
        ]

        finite = values[
            np.isfinite(values)
        ]

        replacement = (
            float(
                finite.max()
            )
            +
            1.0
            if finite.size
            else 1.0
        )

        return np.where(
            np.isfinite(values),
            values,
            replacement,
        )

    def update(
        self,
        client_id,
        reward,
    ):

        if not math.isfinite(
            reward
        ):
            return

        self.counts[
            client_id
        ] += 1

        self.reward_sums[
            client_id
        ] += float(
            reward
        )


# ============================================================
# RANK NORMALIZATION
# ============================================================

def rank_normalize(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return values

    if len(values) == 1:
        return np.ones(
            1,
            dtype=np.float64,
        )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(
        len(values),
        dtype=np.float64,
    )

    ranks[
        order
    ] = np.arange(
        len(values),
        dtype=np.float64,
    )

    return (
        ranks
        /
        (
            len(values)
            -
            1
        )
    )


# ============================================================
# SCORE SEPARATION
# ============================================================

def normalized_score_separation(
    scores,
):

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if len(scores) <= 1:
        return 1.0

    ordered = np.sort(
        scores
    )[::-1]

    gaps = np.abs(
        np.diff(
            ordered
        )
    )

    if len(gaps) == 0:
        return 0.0

    typical_gap = float(
        np.median(
            gaps
        )
    )

    score_range = float(
        ordered[0]
        -
        ordered[-1]
    )

    if score_range <= 1e-12:
        return 0.0

    return float(
        np.clip(
            typical_gap
            /
            score_range,
            0.0,
            1.0,
        )
    )


# ============================================================
# DYNAMIC ACCEPTANCE COUNT
# ============================================================

def choose_dynamic_accept_count(
    risk_utilities,
    uncertainties,
    combined_scores,
    predictor_confidence,
    validation_trend_pp,
    affordable_indices,
):

    affordable_indices = [
        int(i)
        for i
        in affordable_indices
    ]

    if not affordable_indices:

        return (
            0,
            "budget_no_capacity",
            0.0,
        )

    risk = np.asarray(
        [
            risk_utilities[i]
            for i
            in affordable_indices
        ],
        dtype=np.float64,
    )

    uncertainty = np.asarray(
        [
            uncertainties[i]
            for i
            in affordable_indices
        ],
        dtype=np.float64,
    )

    combined = np.asarray(
        [
            combined_scores[i]
            for i
            in affordable_indices
        ],
        dtype=np.float64,
    )

    n_affordable = len(
        affordable_indices
    )

    min_allowed = min(
        BATCS_MIN_ACCEPT,
        n_affordable,
    )

    max_allowed = min(
        BATCS_MAX_ACCEPT,
        n_affordable,
    )

    if max_allowed < BATCS_MIN_ACCEPT:

        return (
            max_allowed,
            "budget_limited",
            0.0,
        )

    # --------------------------------------------------------
    # Ambiguity.
    # --------------------------------------------------------

    separation = normalized_score_separation(
        combined
    )

    ambiguity = float(
        np.clip(
            1.0 - separation,
            0.0,
            1.0,
        )
    )

    if (
        separation
        <=
        BATCS_LOW_SCORE_SEPARATION
    ):

        ambiguity = max(
            ambiguity,
            0.85,
        )

    elif (
        separation
        >=
        BATCS_HIGH_SCORE_SEPARATION
    ):

        ambiguity = min(
            ambiguity,
            0.25,
        )

    # --------------------------------------------------------
    # Uncertainty ratio.
    # --------------------------------------------------------

    mean_abs_risk = float(
        np.mean(
            np.abs(
                risk
            )
        )
    )

    mean_uncertainty = float(
        np.mean(
            np.maximum(
                uncertainty,
                0.0,
            )
        )
    )

    uncertainty_ratio = float(
        np.clip(
            mean_uncertainty
            /
            max(
                mean_abs_risk,
                1e-6,
            ),
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Validation deterioration.
    # --------------------------------------------------------

    deterioration = float(
        np.clip(
            -validation_trend_pp
            /
            max(
                BATCS_BAD_TREND_THRESHOLD_PP,
                1e-6,
            ),
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Predictor confidence.
    # --------------------------------------------------------

    confidence = float(
        np.clip(
            predictor_confidence,
            0.0,
            1.0,
        )
    )

    confidence_penalty = (
        1.0
        -
        confidence
    )

    # --------------------------------------------------------
    # Composite conservatism.
    # --------------------------------------------------------

    conservatism = (
        0.40 * ambiguity
        +
        0.25 * uncertainty_ratio
        +
        0.20 * deterioration
        +
        0.15 * confidence_penalty
    )

    conservatism = float(
        np.clip(
            conservatism,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Convert evidence state into allowed count range.
    # --------------------------------------------------------

    if max_allowed == min_allowed:

        target_count = min_allowed

    else:

        raw_count = (
            min_allowed
            +
            conservatism
            *
            (
                max_allowed
                -
                min_allowed
            )
        )

        target_count = int(
            round(
                raw_count
            )
        )

        target_count = int(
            np.clip(
                target_count,
                min_allowed,
                max_allowed,
            )
        )

    # --------------------------------------------------------
    # Positive-utility safeguard.
    # --------------------------------------------------------

    positive_fraction = float(
        np.mean(
            risk > 0.0
        )
    )

    if (
        positive_fraction >= 0.80
        and
        (
            uncertainty_ratio >= 0.25
            or
            ambiguity >= 0.50
            or
            deterioration >= 0.50
        )
    ):

        target_count = min(
            max_allowed,
            max(
                target_count,
                min_allowed + 1,
            ),
        )

    if deterioration >= 0.50:

        mode = "validation_conservative"

    elif uncertainty_ratio >= 0.40:

        mode = "uncertainty_conservative"

    elif ambiguity >= 0.65:

        mode = "weak_score_separation"

    elif confidence_penalty >= 0.50:

        mode = "predictor_low_confidence"

    elif target_count == min_allowed:

        mode = "strong_evidence_selective"

    else:

        mode = "evidence_adaptive"

    return (
        int(target_count),
        mode,
        float(separation),
    )


# ============================================================
# STAGE-2 DECISION
# ============================================================

@dataclass
class Stage2Decision:

    accepted_positions: tuple
    mode: str

    predicted_utilities: np.ndarray
    uncertainties: np.ndarray
    risk_utilities: np.ndarray

    utility_per_mb: np.ndarray
    combined_scores: np.ndarray

    percentile_threshold: float

    predictor_weight: float
    predictor_confidence: float

    desired_count: int
    score_separation: float

    affordable_count: int
    percentile_eligible_count: int
    dynamic_pool_count: int


def choose_batcs_stage2(
    features,
    update_sizes_mb,
    ucb_scores,
    predictor,
    round_number,
    validation_trend_pp,
    acquisition_budget_mb,
    rng,
):
    """
Stage-2 decision.

Layer-B pipeline:

    preview
        -> learned utility prediction
        -> uncertainty-aware utility
        -> utility per MB
        -> percentile evidence boundary
        -> dynamic acceptance count
        -> controlled borderline exploration
        -> exact budget-feasible subset
        -> aggregation
        -> post-aggregation marginal validation feedback

Important:

1. The percentile threshold identifies the strong-candidate
   region. It is NOT a hard cardinality constraint.

2. The dynamic-count controller operates over the dynamic
   candidate pool of individually affordable Stage-1 clients.

3. The dynamic candidate pool contains the strong percentile
   candidates plus a controlled number of borderline candidates
   immediately below the percentile boundary.

4. Exact subset optimization enforces the requested dynamic count
   whenever a feasible subset exists.

5. The communication budget remains a hard constraint.

6. No post-decision information is used here.
"""

    n = len(features)

    empty = np.asarray(
        [],
        dtype=np.float64,
    )

    if n == 0:

        return Stage2Decision(
            (),
            "no_candidates",
            empty,
            empty,
            empty,
            empty,
            empty,
            np.nan,
            0.0,
            0.0,
            0,
            np.nan,
            0,
            0,
            0,
        )

    update_sizes_mb = np.asarray(
        update_sizes_mb,
        dtype=np.float64,
    )

    ucb_scores = np.asarray(
        ucb_scores,
        dtype=np.float64,
    )

    acquisition_budget_mb = max(
        float(
            acquisition_budget_mb
        ),
        0.0,
    )

    # ========================================================
    # PREDICTOR
    # ========================================================

    predictions = []
    uncertainties = []

    for feature in features:

        prediction, uncertainty = (
            predictor.predict_with_uncertainty(
                feature
            )
        )

        predictions.append(
            float(
                prediction
            )
        )

        uncertainties.append(
            float(
                max(
                    uncertainty,
                    0.0,
                )
            )
        )

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    uncertainties = np.asarray(
        uncertainties,
        dtype=np.float64,
    )

    # Confidence grows conservatively: the predictor must accumulate
    # a substantial history before it can materially override UCB.
    predictor_confidence = min(
        1.0,
        predictor.n_samples
        /
        max(
            2 * BATCS_MIN_PREDICTOR_SAMPLES,
            1,
        ),
    )

    predictor_weight = (
        BATCS_MAX_PREDICTOR_WEIGHT
        *
        predictor_confidence
    )

    # ========================================================
    # UNCERTAINTY-AWARE UTILITY
    # ========================================================

    risk_utilities = (
        predictions
        +
        BATCS_UNCERTAINTY_BETA
        *
        uncertainties
    )

    # ========================================================
    # UTILITY PER ACTUAL SERIALIZED MB
    # ========================================================

    utility_per_mb = (
        risk_utilities
        /
        np.maximum(
            update_sizes_mb,
            1e-12,
        )
    )

    # ========================================================
    # COMBINE PREDICTOR + UCB
    # ========================================================

    predictor_rank = rank_normalize(
        utility_per_mb
    )

    ucb_rank = rank_normalize(
        ucb_scores
    )

    combined_scores = (
        predictor_weight
        *
        predictor_rank
        +
        (
            1.0
            -
            predictor_weight
        )
        *
        ucb_rank
    )

    # ========================================================
    # INDIVIDUALLY AFFORDABLE CLIENTS
    # ========================================================

    affordable_indices = [
        int(i)
        for i in range(n)
        if (
            update_sizes_mb[i]
            <=
            acquisition_budget_mb
            +
            1e-12
        )
    ]

    affordable_count = len(
        affordable_indices
    )

    if not affordable_indices:

        return Stage2Decision(
            (),
            "budget_no_capacity",
            predictions,
            uncertainties,
            risk_utilities,
            utility_per_mb,
            combined_scores,
            np.nan,
            predictor_weight,
            predictor_confidence,
            0,
            np.nan,
            0,
            0,
            0,
        )

    # ========================================================
    # COLD START / PREDICTOR-UNCERTAIN MODE
    # ========================================================

    predictor_uncertain = (
        round_number
        <=
        BATCS_WARMUP_ROUNDS
        or
        predictor.n_samples
        <
        BATCS_MIN_PREDICTOR_SAMPLES
    )

    if predictor_uncertain:

        # During cold start, trust the already-observed UCB history
        # instead of the untrained ridge predictor.  This prevents
        # zero-initialized predictions/uncertainty from steering the
        # first accepted updates.
        ucb_rank = rank_normalize(
            ucb_scores
        )

        warmup_scores = (
            0.85 * ucb_rank
            +
            0.15 * rank_normalize(utility_per_mb)
        )

        ranked = sorted(
            affordable_indices,
            key=lambda i: (
                warmup_scores[i],
                ucb_scores[i],
                utility_per_mb[i],
            ),
            reverse=True,
        )

        desired = min(
            BATCS_SOFT_MAX_ACCEPT,
            len(ranked),
            BATCS_MAX_ACCEPT,
        )

        selected = []
        spent = 0.0

        for position in ranked:

            cost = float(
                update_sizes_mb[position]
            )

            if (
                spent
                +
                cost
                <=
                acquisition_budget_mb
                +
                1e-12
            ):
                selected.append(position)
                spent += cost

            if len(selected) >= desired:
                break

        separation = normalized_score_separation(
            warmup_scores[affordable_indices]
        )

        return Stage2Decision(
            tuple(selected),
            "ucb_protected_warmup",
            predictions,
            uncertainties,
            risk_utilities,
            utility_per_mb,
            combined_scores,
            np.nan,
            predictor_weight,
            predictor_confidence,
            len(selected),
            separation,
            affordable_count,
            affordable_count,
            affordable_count,
        )

    # ========================================================
    # TRUE PERCENTILE THRESHOLD
    # ========================================================

    eligible_scores = utility_per_mb[
    affordable_indices
]

    percentile_threshold = float(
        np.percentile(
            eligible_scores,
            BATCS_PERCENTILE,
        )
    )

    # Strong percentile candidates.
    percentile_eligible_indices = [
        int(i)
        for i in affordable_indices
        if (
            utility_per_mb[i]
            >=
            percentile_threshold
            -
            1e-12
        )
    ]

    percentile_eligible_count = len(
        percentile_eligible_indices
    )

# --------------------------------------------------------
# BORDERLINE CANDIDATES
#
# Keep a small number of candidates immediately below the
# percentile boundary so that dynamic acceptance can
# genuinely choose between 4 and 7 clients.
# --------------------------------------------------------

    below_percentile = [
        int(i)
        for i in affordable_indices
        if (
            utility_per_mb[i]
            <
            percentile_threshold
        )
    ]

    below_percentile = sorted(
        below_percentile,
        key=lambda i: (
            utility_per_mb[i],
            combined_scores[i],
        ),
        reverse=True,
    )

    borderline_candidates = (
        below_percentile[
            :BATCS_BORDERLINE_POOL
        ]
    )

# --------------------------------------------------------
# EXPANDED DYNAMIC-CONTROL POOL
#
# Strong percentile candidates are always included.
# Borderline candidates are included only when they are
# among the highest-ranked candidates immediately below
# the percentile boundary.
# --------------------------------------------------------

    dynamic_candidate_pool = sorted(
        set(
            percentile_eligible_indices
            +
            borderline_candidates
        )
    )

    if not percentile_eligible_indices:

        return Stage2Decision(
            (),
            "percentile_no_candidate",
            predictions,
            uncertainties,
            risk_utilities,
            utility_per_mb,
            combined_scores,
            percentile_threshold,
            predictor_weight,
            predictor_confidence,
            0,
            0.0,
            affordable_count,
            0,
            0,
        )

    # ========================================================
    # SCORE SEPARATION
    # ========================================================

    score_separation = normalized_score_separation(
        combined_scores[
            dynamic_candidate_pool
        ]
    )

    # ========================================================
# DYNAMIC COUNT
#
# The count controller sees the expanded dynamic candidate
# pool: all strong percentile candidates plus the controlled
# borderline candidates, subject to individual affordability.
# ========================================================

    desired_count, count_mode, _ = (
        choose_dynamic_accept_count(
            risk_utilities=risk_utilities,
            uncertainties=uncertainties,
            combined_scores=combined_scores,
            predictor_confidence=predictor_confidence,
            validation_trend_pp=validation_trend_pp,
            affordable_indices=dynamic_candidate_pool,
        )
    )

    # Safety cap: 6-7 clients are allowed only when the learned
    # predictor is mature and the evidence is unusually favorable.
    effective_max_accept = BATCS_SOFT_MAX_ACCEPT

    mean_risk_dynamic = float(
        np.mean(
            risk_utilities[dynamic_candidate_pool]
        )
    )
    mean_uncertainty_dynamic = float(
        np.mean(
            uncertainties[dynamic_candidate_pool]
        )
    )
    uncertainty_to_risk = (
        mean_uncertainty_dynamic
        /
        max(abs(mean_risk_dynamic), 1e-6)
    )

    if (
        predictor_confidence >= 0.75
        and
        validation_trend_pp >= 0.0
        and
        mean_risk_dynamic > 0.0
        and
        uncertainty_to_risk <= 0.50
    ):
        effective_max_accept = BATCS_MAX_ACCEPT

    desired_count = int(
        np.clip(
            desired_count,
            0,
            min(
                effective_max_accept,
                len(dynamic_candidate_pool),
            ),
        )
    )

    if desired_count <= 0:

        return Stage2Decision(
            (),
            "no_dynamic_capacity",
            predictions,
            uncertainties,
            risk_utilities,
            utility_per_mb,
            combined_scores,
            percentile_threshold,
            predictor_weight,
            predictor_confidence,
            0,
            score_separation,
            affordable_count,
            percentile_eligible_count,
            len(dynamic_candidate_pool),
        )

    # ========================================================
# EXPLORATION
#
# Exploration may consider the controlled borderline pool.
# Borderline candidates are restricted to the explicitly
# retained candidates immediately below the percentile boundary.
# ========================================================

    dynamic_ranked = sorted(
        dynamic_candidate_pool,
        key=lambda i: (
            utility_per_mb[i],
            combined_scores[i],
        ),
        reverse=True,
    )

    selected_seed = dynamic_ranked[
        :desired_count
    ].copy()

    if (
        selected_seed
        and
        predictor.n_samples
        >=
        BATCS_MIN_PREDICTOR_SAMPLES
        and
        rng.random()
        <
        BATCS_EXPLORATION_PROB
    ):

        rejected = [
            i
            for i
            in dynamic_ranked
            if i
            not in
            set(selected_seed)
        ]

        if rejected:

            # Replace with a controlled borderline candidate
            # retained immediately below the percentile boundary.
            # Exploration remains restricted to the explicit borderline pool.
            boundary_position = (
                selected_seed[-1]
            )

            boundary_score = float(
                utility_per_mb[
                    boundary_position
                ]
            )

            borderline = [
                i
                for i
                in rejected
                if (
                    i in borderline_candidates
                    and
            utility_per_mb[i]
            >=
            0.90
            *
            max(
                boundary_score,
                1e-12,
                )
            )
        ]

            if borderline:

                selected_seed[-1] = (
                    borderline[0]
                )

    # ========================================================
# EXACT SUBSET OPTIMIZATION
#
# Search the complete dynamic candidate pool:
#
#     strong percentile candidates
#     +
#     controlled borderline candidates
#
# No additional top-k truncation is applied.
#
# At most 10 Stage-1 candidates -> at most 2^10 = 1024 subsets.
# ========================================================

    optimization_pool = sorted(
        set(
            dynamic_candidate_pool
        )
    )

    pool_n = len(
        optimization_pool
    )

    # Search desired count first, then the largest smaller count.
    target_counts = list(
        range(
            min(
                desired_count,
                pool_n,
            ),
            0,
            -1,
        )
    )

    best_subset = None
    best_key = None
    best_count = 0

    # --------------------------------------------------------
    # Exact search.
    # --------------------------------------------------------

    for target_count in target_counts:

        count_best_subset = None
        count_best_key = None

        for mask in range(
            1 << pool_n
        ):

            subset = [
                optimization_pool[j]
                for j in range(
                    pool_n
                )
                if (
                    mask
                    >>
                    j
                )
                &
                1
            ]

            if len(subset) != target_count:
                continue

            subset_cost = float(
                sum(
                    update_sizes_mb[i]
                    for i
                    in subset
                )
            )

            if (
                subset_cost
                >
                acquisition_budget_mb
                +
                1e-12
            ):
                continue


            total_risk = float(
                sum(
                    risk_utilities[i]
                    for i
                    in subset
                )
            )

            total_combined = float(
                sum(
                    combined_scores[i]
                    for i
                    in subset
                )
            )

            total_utility_per_mb = float(
                sum(
                    utility_per_mb[i]
                    for i
                    in subset
                )
            )

            total_uncertainty = float(
                sum(
                    uncertainties[i]
                    for i
                    in subset
                )
            )

            # Maximize utility first.
            # Ranking/efficiency/uncertainty are secondary tie-breakers.
            total_ucb = float(
                sum(
                    ucb_scores[i]
                    for i
                    in subset
                )
            )

            key = (
                total_risk,
                total_combined,
                total_ucb,
                total_utility_per_mb,
                -total_uncertainty,
                -subset_cost,
            )

            if (
                count_best_key is None
                or
                key
                >
                count_best_key
            ):

                count_best_key = key
                count_best_subset = subset

        if count_best_subset is not None:

            best_subset = (
                count_best_subset
            )

            best_key = (
                count_best_key
            )

            best_count = (
                target_count
            )

            # Desired count satisfied exactly.
            break

    # --------------------------------------------------------
    # If desired count is impossible under actual byte cost,
    # choose the largest feasible count below desired count.
    # --------------------------------------------------------

    if best_subset is None:

        for target_count in range(
            min(
                desired_count - 1,
                pool_n,
            ),
            0,
            -1,
        ):

            count_best_subset = None
            count_best_key = None

            for mask in range(
                1 << pool_n
            ):

                subset = [
                    optimization_pool[j]
                    for j in range(
                        pool_n
                    )
                    if (
                        mask
                        >>
                        j
                    )
                    &
                    1
                ]

                if len(subset) != target_count:
                    continue

                subset_cost = float(
                    sum(
                        update_sizes_mb[i]
                        for i
                        in subset
                    )
                )

                if (
                    subset_cost
                    >
                    acquisition_budget_mb
                    +
                    1e-12
                ):
                    continue

                total_risk = float(
                    sum(
                        risk_utilities[i]
                        for i
                        in
                        subset
                    )
                )

                total_combined = float(
                    sum(
                        combined_scores[i]
                        for i
                        in
                        subset
                    )
                )

                total_utility_per_mb = float(
                    sum(
                        utility_per_mb[i]
                        for i
                        in
                        subset
                    )
                )

                total_uncertainty = float(
                    sum(
                        uncertainties[i]
                        for i
                        in
                        subset
                    )
                )

                total_ucb = float(
                    sum(
                        ucb_scores[i]
                        for i
                        in
                        subset
                    )
                )

                key = (
                    total_risk,
                    total_combined,
                    total_ucb,
                    total_utility_per_mb,
                    -total_uncertainty,
                    -subset_cost,
                )

                if (
                    count_best_key is None
                    or
                    key
                    >
                    count_best_key
                ):

                    count_best_key = key
                    count_best_subset = subset

            if count_best_subset is not None:

                best_subset = (
                    count_best_subset
                )

                best_key = (
                    count_best_key
                )

                best_count = (
                    target_count
                )

                break

   # Final safety fallback:
# best single feasible candidate from the dynamic candidate pool.

    if best_subset is None:

        ranked = sorted(
            optimization_pool,
            key=lambda i: (
                utility_per_mb[i],
                combined_scores[i],
            ),
            reverse=True,   
        )

        for i in ranked:

            if (
                update_sizes_mb[i]
                <=
                acquisition_budget_mb
                +
                1e-12
            ):

                best_subset = [
                    i
                ]

                best_count = 1

                break

    if best_subset is None:

        best_subset = []
        best_count = 0

    # ========================================================
    # CONTROLLED DYNAMIC-COUNT ASSERTION
    # ========================================================

    if best_subset:

        if len(
            best_subset
        ) > BATCS_MAX_ACCEPT:

            raise RuntimeError(
                "BATCS accepted more than BATCS_MAX_ACCEPT."
            )

    # ========================================================
    # FINAL MODE
    # ========================================================

    if (
        len(best_subset)
        <
        desired_count
    ):

        final_mode = (
            "budget_limited"
        )

    elif (
        validation_trend_pp
        <
        -BATCS_BAD_TREND_THRESHOLD_PP
    ):

        final_mode = (
            "validation_trend_safeguard"
        )

    elif (
        np.mean(
            uncertainties[
                dynamic_candidate_pool
            ]
        )
        /
        max(
            np.mean(
                np.abs(
                    predictions[
                        dynamic_candidate_pool
                    ]
                )
            ),
            1e-6,
        )
        >=
        0.40
    ):

        final_mode = (
            "uncertainty_conservative"
        )

    elif (
        score_separation
        <=
        BATCS_LOW_SCORE_SEPARATION
    ):

        final_mode = (
            "weak_score_separation"
        )

    else:

        final_mode = count_mode

    return Stage2Decision(
        tuple(
            int(i)
            for i
            in best_subset
        ),
        final_mode,
        predictions,
        uncertainties,
        risk_utilities,
        utility_per_mb,
        combined_scores,
        percentile_threshold,
        predictor_weight,
        predictor_confidence,
        best_count,
        score_separation,
        affordable_count,
        percentile_eligible_count,
        len(dynamic_candidate_pool),
    )


# ============================================================
# AGGREGATE-MARGINAL VALIDATION UTILITY
# ============================================================

def aggregate_validation_utility(
    global_state,
    local_states,
    accepted_positions,
    client_weights,
    validation_loader,
    validation_model,
):
    """
    Compute aggregate-marginal validation utility AFTER the
    current Stage-2 decision and AFTER the accepted aggregate
    has been formed.

    For an accepted client i:

        u_i =
            A(G_S)
            -
            A(G_{S \\ i})

    where:
        S = actually accepted clients in the current round
        G_S = FedAvg of all accepted updates
        G_{S\\i} = FedAvg of all accepted updates except i

    For the single-client case:

        u_i =
            A(G_i)
            -
            A(G_global_before)

    The result is therefore an actual marginal contribution to
    the accepted aggregate, not isolated client quality.

    This function must ONLY be called after Stage-2 selection.
    """

    if not accepted_positions:
        return {}

    accepted_positions = [
        int(position)
        for position
        in accepted_positions
    ]

    accepted_states = [
        local_states[position]
        for position
        in accepted_positions
    ]

    accepted_weights = [
        client_weights[position]
        for position
        in accepted_positions
    ]

    # --------------------------------------------------------
    # Final accepted aggregate.
    # --------------------------------------------------------

    final_aggregate_state = fedavg_aggregate(
        accepted_states,
        accepted_weights,
    )

    validation_model.load_state_dict(
        final_aggregate_state,
        strict=True,
    )

    final_aggregate_accuracy = (
        evaluate_accuracy(
            validation_model,
            validation_loader,
            DEVICE,
        )
    )

    utilities = {}

    # --------------------------------------------------------
    # Marginal contribution.
    # --------------------------------------------------------

    for position in accepted_positions:

        if len(
            accepted_positions
        ) == 1:

            # Removing the only update leaves the server at the
            # pre-round global state.
            validation_model.load_state_dict(
                global_state,
                strict=True,
            )

            without_i_accuracy = (
                evaluate_accuracy(
                    validation_model,
                    validation_loader,
                    DEVICE,
                )
            )

            utilities[position] = float(
                final_aggregate_accuracy
                -
                without_i_accuracy
            )

            continue

        remaining_positions = [
            other
            for other
            in accepted_positions
            if other != position
        ]

        remaining_states = [
            local_states[other]
            for other
            in remaining_positions
        ]

        remaining_weights = [
            client_weights[other]
            for other
            in remaining_positions
        ]

        without_i_state = fedavg_aggregate(
            remaining_states,
            remaining_weights,
        )

        validation_model.load_state_dict(
            without_i_state,
            strict=True,
        )

        without_i_accuracy = (
            evaluate_accuracy(
                validation_model,
                validation_loader,
                DEVICE,
            )
        )

        utilities[position] = float(
            final_aggregate_accuracy
            -
            without_i_accuracy
        )

    return utilities


# ============================================================
# BUDGET
# ============================================================

def calculate_budget(
    model_mb,
):

    # LAYER C — unchanged.
    #
    # B0 =
    #   full-horizon mandatory downlink
    #   +
    #   MAIN_BUDGET_RATIO × full-horizon uplink envelope

    downlink_envelope = (
        NUM_ROUNDS
        *
        CLIENTS_PER_ROUND
        *
        model_mb
    )

    full_uplink_envelope = (
        NUM_ROUNDS
        *
        CLIENTS_PER_ROUND
        *
        model_mb
    )

    fedavg_envelope = (
        downlink_envelope
        +
        full_uplink_envelope
    )

    hard_budget = (
        downlink_envelope
        +
        MAIN_BUDGET_RATIO
        *
        full_uplink_envelope
    )

    return (
        hard_budget,
        fedavg_envelope,
    )


# ============================================================
# COMMUNICATION LEDGER
# ============================================================

class CommunicationLedger:

    def __init__(
        self,
        budget_mb,
    ):

        self.budget_mb = float(
            budget_mb
        )

        self.used_mb = 0.0

        self.rows = []

    def remaining(
        self,
    ):

        if not math.isfinite(
            self.budget_mb
        ):

            return float("inf")

        return max(
            self.budget_mb
            -
            self.used_mb,
            0.0,
        )

    def can_charge(
        self,
        amount_mb,
    ):

        if not math.isfinite(
            self.budget_mb
        ):

            return True

        return (
            self.used_mb
            +
            float(amount_mb)
            <=
            self.budget_mb
            +
            1e-9
        )

    def charge(
        self,
        round_number,
        selected_count,
        accepted_mask,
        downlink_mb,
        preview_mb,
        update_sizes_mb,
        method,
        test_accuracy,
        validation_before,
        validation_trend_pp,
        stage2_mode,
    ):

        accepted_mask = np.asarray(
            accepted_mask,
            dtype=bool,
        )

        update_sizes_mb = np.asarray(
            update_sizes_mb,
            dtype=np.float64,
        )

        if len(
            accepted_mask
        ) != selected_count:

            raise RuntimeError(
                "Accepted-mask size mismatch."
            )

        accepted_full_mb = float(
            update_sizes_mb[
                accepted_mask
            ].sum()
        )

        rejected_full_mb = float(
            update_sizes_mb[
                ~accepted_mask
            ].sum()
        )

        round_total = (
            float(downlink_mb)
            +
            float(preview_mb)
            +
            accepted_full_mb
        )

        if not self.can_charge(
            round_total
        ):

            raise RuntimeError(
                f"Round {round_number}: communication budget exceeded."
            )

        self.used_mb += (
            round_total
        )

        rejected_count = (
            selected_count
            -
            int(
                accepted_mask.sum()
            )
        )

        net_saving = (
            rejected_full_mb
            -
            float(preview_mb)
        )

        row = {
            "Round":
                int(
                    round_number
                ),

            "Method":
                method,

            "Selected":
                int(
                    selected_count
                ),

            "Accepted":
                int(
                    accepted_mask.sum()
                ),

            "Rejected":
                int(
                    rejected_count
                ),

            "Downlink_MB":
                float(
                    downlink_mb
                ),

            "Preview_MB":
                float(
                    preview_mb
                ),

            "Accepted_Full_MB":
                accepted_full_mb,

            "Rejected_Avoided_MB":
                rejected_full_mb,

            "Net_Saving_MB":
                net_saving,

            "Round_Communication_MB":
                round_total,

            "Cumulative_Communication_MB":
                float(
                    self.used_mb
                ),

            "Remaining_Budget_MB":
                float(
                    self.remaining()
                ),

            "Test_Accuracy_pct":
                float(
                    test_accuracy
                ),

            "Validation_Before_pct":
                (
                    float(
                        validation_before
                    )
                    if math.isfinite(
                        validation_before
                    )
                    else np.nan
                ),

            "Validation_Trend_pp":
                float(
                    validation_trend_pp
                ),

            "Stage2_Mode":
                stage2_mode,
        }

        self.rows.append(
            row
        )

        return row


# ============================================================
# RUN ONE METHOD / ONE SEED
# ============================================================

def run_method(
    method,
    seed,
    train_set,
    train_eval_set,
    test_loader,
    client_indices,
    validation_indices,
    model_mb,
    budget_mb,
):

    set_seed(seed)

    # --------------------------------------------------------
    # Client loaders
    # --------------------------------------------------------

    client_loaders = [
        DataLoader(
            Subset(
                train_set,
                indices.tolist(),
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        for indices
        in client_indices
    ]

    # --------------------------------------------------------
    # Server validation loader
    # --------------------------------------------------------

    server_val_loader = DataLoader(
        Subset(
            train_eval_set,
            validation_indices.tolist(),
        ),
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    global_model = CIFAR10CNN().to(
        DEVICE
    )

    global_state = clone_state_to_cpu(
        global_model.state_dict()
    )

    worker_model = CIFAR10CNN().to(
        DEVICE
    )

    # Reused validation model:
    # avoids repeated model allocations on MPS.
    validation_model = CIFAR10CNN().to(
        DEVICE
    )

    # --------------------------------------------------------
    # Policies
    # --------------------------------------------------------

    ucb = (
        UCB1(
            NUM_CLIENTS,
            seed,
        )
        if method
        in {
            "UCB",
            "UCB+Budget",
            "BATCS",
        }
        else None
    )

    predictor = (
        OnlineRidgePredictor(
            num_features=5,
            regularization=1.0,
        )
        if method == "BATCS"
        else None
    )

    ledger = CommunicationLedger(
        budget_mb
        if method
        in {
            "Random+Budget",
            "UCB+Budget",
            "BATCS",
        }
        else math.inf
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    round_records = []
    predictor_records = []

    previous_validation_accuracy = None

    run_start = time.perf_counter()

    initial_test_accuracy = evaluate_accuracy(
        global_model,
        test_loader,
        DEVICE,
    )

    # ========================================================
    # ROUNDS
    # ========================================================

    for round_number in range(
        1,
        NUM_ROUNDS + 1,
    ):

        round_start = time.perf_counter()

        # ====================================================
        # STAGE 1 — FIXED 10 / 50
        # ====================================================

        if method in {
            "UCB",
            "UCB+Budget",
            "BATCS",
        }:

            selected = ucb.select(
                CLIENTS_PER_ROUND
            )

        else:

            selection_rng = np.random.default_rng(
                seed
                +
                round_number
                +
                10_000
            )

            selected = (
                selection_rng.choice(
                    NUM_CLIENTS,
                    size=CLIENTS_PER_ROUND,
                    replace=False,
                )
                .astype(int)
                .tolist()
            )

        if len(
            selected
        ) != CLIENTS_PER_ROUND:

            raise RuntimeError(
                "Stage-1 selected client count changed."
            )

        # ====================================================
        # MANDATORY DOWNLINK
        # ====================================================

        downlink_mb = (
            CLIENTS_PER_ROUND
            *
            model_mb
        )

        if not ledger.can_charge(
            downlink_mb
        ):

            print(
                f"{method:16s} "
                f"seed={seed} "
                f"budget exhausted before "
                f"round {round_number}."
            )

            break

        # ====================================================
        # BATCS PRE-ROUND VALIDATION
        # ====================================================

        if method == "BATCS":

            validation_model.load_state_dict(
                global_state,
                strict=True,
            )

            validation_before = (
                evaluate_accuracy(
                    validation_model,
                    server_val_loader,
                    DEVICE,
                )
            )

            if previous_validation_accuracy is None:

                validation_trend_pp = 0.0

            else:

                validation_trend_pp = (
                    validation_before
                    -
                    previous_validation_accuracy
                )

        else:

            validation_before = np.nan
            validation_trend_pp = 0.0

        # ====================================================
        # LOCAL TRAINING
        # ====================================================

        local_states = []
        local_losses = []
        first_losses = []
        client_sizes = []

        for client_id in selected:

            state, avg_loss, first_loss = (
                local_train(
                    worker_model,
                    global_state,
                    client_loaders[
                        client_id
                    ],
                )
            )

            local_states.append(
                state
            )

            local_losses.append(
                avg_loss
            )

            first_losses.append(
                first_loss
            )

            client_sizes.append(
                len(
                    client_indices[
                        client_id
                    ]
                )
            )

        # ====================================================
        # ACTUAL SERIALIZED UPDATE SIZES
        # ====================================================

        full_sizes_mb = np.asarray(
            [
                serialized_size_mb(
                    state
                )
                for state
                in local_states
            ],
            dtype=np.float64,
        )

        # ====================================================
        # STAGE 2
        # ====================================================

        preview_mb = 0.0
        decision = None

        if method == "BATCS":

            previews = [
                compute_preview_stats(
                    global_state,
                    local_state,
                    avg_loss,
                )
                for local_state, avg_loss
                in zip(
                    local_states,
                    local_losses,
                )
            ]

            preview_sizes_mb = np.asarray(
                [
                    preview_size_mb(
                        preview
                    )
                    for preview
                    in previews
                ],
                dtype=np.float64,
            )

            preview_mb = float(
                preview_sizes_mb.sum()
            )

            remaining_after_mandatory = (
                ledger.remaining()
                -
                downlink_mb
                -
                preview_mb
            )

            if remaining_after_mandatory < 0.0:

                print(
                    "BATCS cannot afford complete "
                    f"preview transmission at round {round_number}."
                )

                break

            # ------------------------------------------------
            # Reserve future mandatory downlink + preview.
            # Full updates remain the controllable acquisition
            # component.
            # ------------------------------------------------

            future_rounds = (
                NUM_ROUNDS
                -
                round_number
            )

            mean_preview_mb = float(
                np.mean(
                    preview_sizes_mb
                )
            )

            future_mandatory_reserve = (
                future_rounds
                *
                (
                    downlink_mb
                    +
                    mean_preview_mb
                    *
                    CLIENTS_PER_ROUND
                )
            )

            acquisition_budget_mb = max(
                0.0,
                remaining_after_mandatory
                -
                future_mandatory_reserve,
            )

            features = [
                preview_to_features(
                    preview,
                    full_sizes_mb[position],
                )
                for position, preview
                in enumerate(
                    previews
                )
            ]

            ucb_snapshot = ucb.scores_for(
                selected
            )

            decision = choose_batcs_stage2(
                features=features,
                update_sizes_mb=full_sizes_mb,
                ucb_scores=ucb_snapshot,
                predictor=predictor,
                round_number=round_number,
                validation_trend_pp=validation_trend_pp,
                acquisition_budget_mb=acquisition_budget_mb,
                rng=np.random.default_rng(
                    seed
                    +
                    round_number
                    +
                    20_000
                ),
            )

            accepted_positions = list(
                decision.accepted_positions
            )

            stage2_mode = decision.mode

        elif method == "UCB+Budget":

            # Layer A/C unchanged:
            # no Stage-2 filtering.
            accepted_positions = list(
                range(
                    CLIENTS_PER_ROUND
                )
            )

            stage2_mode = (
                "all_stage1_selected"
            )

        elif method == "Random+Budget":

            accepted_positions = list(
                range(
                    CLIENTS_PER_ROUND
                )
            )

            stage2_mode = (
                "all_stage1_selected"
            )

        else:

            accepted_positions = list(
                range(
                    CLIENTS_PER_ROUND
                )
            )

            stage2_mode = (
                "no_stage2"
            )

        # ====================================================
        # COMMUNICATION FEASIBILITY
        # ====================================================

        accepted_mask = np.zeros(
            CLIENTS_PER_ROUND,
            dtype=bool,
        )

        accepted_mask[
            accepted_positions
        ] = True

        accepted_full_mb = float(
            full_sizes_mb[
                accepted_mask
            ].sum()
        )

        round_total = (
            downlink_mb
            +
            preview_mb
            +
            accepted_full_mb
        )

        if not ledger.can_charge(
            round_total
        ):

            if method != "BATCS":

                print(
                    f"{method:16s} "
                    f"seed={seed} "
                    f"cannot fund complete "
                    f"round {round_number}. "
                    f"Stopping cleanly."
                )

                break

            # ------------------------------------------------
            # Final BATCS exact-feasibility recovery.
            #
            # IMPORTANT:
            # Recovery is restricted to the already-accepted
            # dynamic candidate set and removes the weakest
            # accepted candidate first.
            # ------------------------------------------------

            if decision is None:

                raise RuntimeError(
                    "BATCS decision missing."
                )

            ranked_accepted = sorted(
                accepted_positions,
                key=lambda position: (
                    decision.utility_per_mb[
                        position
                    ],
                    decision.combined_scores[
                        position
                    ],
                ),
            )

            while ranked_accepted:

                candidate_mask = np.zeros(
                    CLIENTS_PER_ROUND,
                    dtype=bool,
                )

                candidate_mask[
                    ranked_accepted
                ] = True

                candidate_full_mb = float(
                    full_sizes_mb[
                        candidate_mask
                    ].sum()
                )

                candidate_total = (
                    downlink_mb
                    +
                    preview_mb
                    +
                    candidate_full_mb
                )

                if ledger.can_charge(
                    candidate_total
                ):

                    accepted_mask = (
                        candidate_mask
                    )

                    accepted_positions = (
                        ranked_accepted.copy()
                    )

                    round_total = (
                        candidate_total
                    )

                    break

                ranked_accepted.pop()

            else:

                print(
                    "BATCS has no affordable full update "
                    f"for round {round_number}."
                )

                break

        # ====================================================
        # COMMUNICATION CHARGE
        # ====================================================

        row = ledger.charge(
            round_number=round_number,
            selected_count=len(
                selected
            ),
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

        # ====================================================
        # SAVE PRE-AGGREGATION GLOBAL STATE
        #
        # Needed so the one-client marginal target can compare
        # against the actual pre-round global model.
        # ====================================================

        pre_aggregation_global_state = (
            {
                key:
                    value.clone()
                for key, value
                in global_state.items()
            }
        )

        # ====================================================
        # AGGREGATION
        # ====================================================

        if accepted_positions:

            accepted_states = [
                local_states[
                    position
                ]
                for position
                in accepted_positions
            ]

            accepted_weights = [
                client_sizes[
                    position
                ]
                for position
                in accepted_positions
            ]

            aggregated_state = (
                fedavg_aggregate(
                    accepted_states,
                    accepted_weights,
                )
            )

            global_state = (
                aggregated_state
            )

            global_model.load_state_dict(
                global_state,
                strict=True,
            )

        else:

            global_model.load_state_dict(
                global_state,
                strict=True,
            )

        # ====================================================
        # POST-AGGREGATION FEEDBACK
        #
        # IMPORTANT FINAL PATCH:
        #
        # The predictor target is now computed AFTER the actual
        # accepted aggregate exists.
        # ====================================================

        if method == "BATCS":

            accepted_set = set(
                accepted_positions
            )

            aggregate_utilities = (
                aggregate_validation_utility(
                    global_state=(
                        pre_aggregation_global_state
                    ),
                    local_states=local_states,
                    accepted_positions=accepted_positions,
                    client_weights=client_sizes,
                    validation_loader=server_val_loader,
                    validation_model=validation_model,
                )
            )

            for position in range(
                CLIENTS_PER_ROUND
            ):

                accepted_flag = (
                    position
                    in
                    accepted_set
                )

                # ------------------------------------------------
                # Diagnostic isolated-client validation.
                # This is NEVER used as the predictor target.
                # ------------------------------------------------

                validation_model.load_state_dict(
                    local_states[
                        position
                    ],
                    strict=True,
                )

                isolated_accuracy = (
                    evaluate_accuracy(
                        validation_model,
                        server_val_loader,
                        DEVICE,
                    )
                )

                isolated_utility = (
                    isolated_accuracy
                    -
                    validation_before
                )

                if accepted_flag:

                    observed_marginal_utility = (
                        aggregate_utilities[
                            position
                        ]
                    )

                else:

                    observed_marginal_utility = (
                        np.nan
                    )

                client_comm_mb = (
                    model_mb
                    +
                    float(
                        preview_sizes_mb[
                            position
                        ]
                    )
                    +
                    (
                        float(
                            full_sizes_mb[
                                position
                            ]
                        )
                        if accepted_flag
                        else 0.0
                    )
                )

                # ------------------------------------------------
                # UCB update occurs only AFTER the Stage-2
                # decision and uses the aggregate-marginal target.
                # ------------------------------------------------

                if accepted_flag:

                    reward = (
                        observed_marginal_utility
                        /
                        max(
                            client_comm_mb,
                            1e-12,
                        )
                    )

                    ucb.update(
                        selected[
                            position
                        ],
                        reward,
                    )

                # ------------------------------------------------
                # Predictor diagnostics.
                # ------------------------------------------------

                predictor_records.append(
                    {
                        "Seed":
                            seed,

                        "Round":
                            round_number,

                        "Client":
                            selected[
                                position
                            ],

                        "Predicted_Utility_pp":
                            float(
                                decision.predicted_utilities[
                                    position
                                ]
                            ),

                        "Uncertainty_pp":
                            float(
                                decision.uncertainties[
                                    position
                                ]
                            ),

                        "Risk_Utility_pp":
                            float(
                                decision.risk_utilities[
                                    position
                                ]
                            ),

                        "Utility_per_MB":
                            float(
                                decision.utility_per_mb[
                                    position
                                ]
                            ),

                        "Combined_Score":
                            float(
                                decision.combined_scores[
                                    position
                                ]
                            ),

                        "Percentile_Threshold":
                            float(
                                decision.percentile_threshold
                            ),

                        "Percentile_Eligible":
                            bool(
                                decision.utility_per_mb[
                                    position
                                ]
                                >=
                                decision.percentile_threshold
                            ),

                        "Accepted":
                            bool(
                                accepted_flag
                            ),

                        "Observed_Marginal_Utility_pp":
                            (
                                float(
                                    observed_marginal_utility
                                )
                                if accepted_flag
                                else np.nan
                            ),

                        "Isolated_Utility_pp":
                            float(
                                isolated_utility
                            ),

                        "False_Rejection":
                            bool(
                                (
                                    not accepted_flag
                                )
                                and
                                (
                                    isolated_utility
                                    >
                                    0.0
                                )
                            ),
                    }
                )

            # ------------------------------------------------
            # Predictor is trained ONLY after current-round
            # decision + aggregation + marginal validation.
            # ------------------------------------------------

            for position in accepted_positions:

                predictor.update(
                    features[
                        position
                    ],
                    aggregate_utilities[
                        position
                    ],
                )

            # ------------------------------------------------
            # Validation AFTER actual global aggregation.
            # ------------------------------------------------

            validation_model.load_state_dict(
                global_state,
                strict=True,
            )

            validation_after = (
                evaluate_accuracy(
                    validation_model,
                    server_val_loader,
                    DEVICE,
                )
            )

            previous_validation_accuracy = (
                validation_after
            )

        elif method in {
            "UCB",
            "UCB+Budget",
        }:

            # Baseline validation feedback.
            validation_model.load_state_dict(
                pre_aggregation_global_state,
                strict=True,
            )

            baseline_validation = (
                evaluate_accuracy(
                    validation_model,
                    server_val_loader,
                    DEVICE,
                )
            )

            for position in accepted_positions:

                validation_model.load_state_dict(
                    local_states[
                        position
                    ],
                    strict=True,
                )

                client_validation_accuracy = (
                    evaluate_accuracy(
                        validation_model,
                        server_val_loader,
                        DEVICE,
                    )
                )

                observed_utility = (
                    client_validation_accuracy
                    -
                    baseline_validation
                )

                client_comm_mb = (
                    model_mb
                    +
                    float(
                        full_sizes_mb[
                            position
                        ]
                    )
                )

                reward = (
                    observed_utility
                    /
                    max(
                        client_comm_mb,
                        1e-12,
                    )
                )

                ucb.update(
                    selected[
                        position
                    ],
                    reward,
                )

        # ====================================================
        # TEST ACCURACY
        # ====================================================

        test_accuracy = evaluate_accuracy(
            global_model,
            test_loader,
            DEVICE,
        )

        row[
            "Test_Accuracy_pct"
        ] = test_accuracy

        # ====================================================
        # BATCS DIAGNOSTICS
        # ====================================================

        if method == "BATCS":

            row[
                "Predictor_Samples"
            ] = predictor.n_samples

            row[
                "Predictor_Confidence"
            ] = (
                decision.predictor_confidence
                if decision is not None
                else np.nan
            )

            row[
                "Stage2_Percentile_Threshold"
            ] = (
                decision.percentile_threshold
                if decision is not None
                else np.nan
            )
            row[
                "Stage2_Dynamic_Pool_Count"
            ] = (
                decision.dynamic_pool_count
                if decision is not None
                else np.nan
            )

            row[
                "Stage2_Desired_Count"
            ] = (
                decision.desired_count
                if decision is not None
                else np.nan
            )

            row[
                "Stage2_Score_Separation"
            ] = (
                decision.score_separation
                if decision is not None
                else np.nan
            )

            row[
                "Stage2_Affordable_Count"
            ] = (
                decision.affordable_count
                if decision is not None
                else np.nan
            )

            row[
                "Stage2_Percentile_Eligible_Count"
            ] = (
                decision.percentile_eligible_count
                if decision is not None
                else np.nan
            )

        else:

            row[
                "Predictor_Samples"
            ] = np.nan

            row[
                "Predictor_Confidence"
            ] = np.nan

            row[
                "Stage2_Percentile_Threshold"
            ] = np.nan

            row[
                "Stage2_Desired_Count"
            ] = np.nan

            row[
                "Stage2_Score_Separation"
            ] = np.nan

            row[
                "Stage2_Affordable_Count"
            ] = np.nan

            row[
                "Stage2_Percentile_Eligible_Count"
            ] = np.nan

            row[
                "Stage2_Dynamic_Pool_Count"
            ] = np.nan

        row[
            "Budget_Limit_MB"
        ] = (
            budget_mb
            if math.isfinite(
                ledger.budget_mb
            )
            else np.inf
        )

        row[
            "Initial_Test_Accuracy_pct"
        ] = initial_test_accuracy

        row[
            "Round_Time_sec"
        ] = (
            time.perf_counter()
            -
            round_start
        )

        round_records.append(
            dict(row)
        )

        # ====================================================
        # PRINT
        # ====================================================

        print(
            f"{method:16s} "
            f"seed={seed} "
            f"round={round_number:02d}/{NUM_ROUNDS} "
            f"selected={len(selected):2d} "
            f"accepted={len(accepted_positions):2d} "
            f"acc={test_accuracy:6.2f}% "
            f"comm={ledger.used_mb:8.2f} MB"
        )

        if method == "BATCS":

            desired_text = (
                decision.desired_count
                if decision is not None
                else 0
            )

            eligible_text = (
                decision.percentile_eligible_count
                if decision is not None
                else 0
            )

            threshold_text = (
                decision.percentile_threshold
                if decision is not None
                and
                math.isfinite(
                    decision.percentile_threshold
                )
                else 0.0
            )

            separation_text = (
                decision.score_separation
                if decision is not None
                and
                math.isfinite(
                    decision.score_separation
                )
                else 0.0
            )

            print(
                f"    Stage2={stage2_mode} "
                f"desired={desired_text} "
                f"strong={eligible_text} "
                f"dynamic_pool={decision.dynamic_pool_count if decision is not None else 0} "
                f"accepted={len(accepted_positions)} "
                f"percentile={BATCS_PERCENTILE:.1f} "
                f"threshold={threshold_text:.5f} "
                f"separation={separation_text:.3f}"
            )

            print(
                f"    BATCS BUDGET: "
                f"Downlink={downlink_mb:.4f} MB + "
                f"Preview={preview_mb:.4f} MB + "
                f"AcceptedFull={accepted_full_mb:.4f} MB = "
                f"{round_total:.4f} MB"
            )

        # ====================================================
        # HARD BUDGET STOP
        # ====================================================

        if (
            math.isfinite(
                ledger.budget_mb
            )
            and
            ledger.remaining()
            <
            model_mb
        ):

            print(
                f"    {method}: "
                "budget cannot fund another "
                "complete Stage-1 downlink."
            )

            break

    # ========================================================
    # SUMMARY
    # ========================================================

    if round_records:

        final_accuracy = float(
            round_records[
                -1
            ][
                "Test_Accuracy_pct"
            ]
        )

    else:

        final_accuracy = float(
            initial_test_accuracy
        )

    rounds_completed = len(
        round_records
    )

    total_mb = float(
        ledger.used_mb
    )

    if method in {
        "Random+Budget",
        "UCB+Budget",
        "BATCS",
    }:

        communication_reduction = (
            100.0
            *
            (
                1.0
                -
                total_mb
                /
                max(
                    fedavg_envelope_mb_global,
                    1e-12,
                )
            )
        )

    else:

        communication_reduction = 0.0

    elapsed = (
        time.perf_counter()
        -
        run_start
    )

    summary = {
        "method":
            method,

        "seed":
            seed,

        "final_accuracy":
            final_accuracy,

        "rounds_completed":
            rounds_completed,

        "total_mb":
            total_mb,

        "communication_reduction_pct":
            communication_reduction,

        "accuracy_per_mb":
            (
                final_accuracy
                /
                max(
                    total_mb,
                    1e-12,
                )
            ),

        "elapsed_seconds":
            elapsed,

        "budget_mb":
            (
                budget_mb
                if math.isfinite(
                    budget_mb
                )
                else np.inf
            ),
    }

    return (
        summary,
        round_records,
        predictor_records,
    )


# ============================================================
# TABLE
# ============================================================

def make_table(
    summary_df,
):

    rows = []

    for method in [
        "FedAvg",
        "Random+Budget",
        "UCB",
        "UCB+Budget",
        "BATCS",
    ]:

        subset = summary_df[
            summary_df[
                "method"
            ]
            ==
            method
        ]

        accuracy_mean = (
            subset[
                "final_accuracy"
            ].mean()
        )

        accuracy_std = (
            subset[
                "final_accuracy"
            ].std(
                ddof=1
            )
            if len(subset) > 1
            else 0.0
        )

        mb_mean = (
            subset[
                "total_mb"
            ].mean()
        )

        mb_std = (
            subset[
                "total_mb"
            ].std(
                ddof=1
            )
            if len(subset) > 1
            else 0.0
        )

        reduction_mean = (
            subset[
                "communication_reduction_pct"
            ].mean()
        )

        acc_per_mb = (
            accuracy_mean
            /
            max(
                mb_mean,
                1e-12,
            )
        )

        rows.append(
            {
                "Method":
                    method,

                "Accuracy":
                    (
                        f"{accuracy_mean:.2f} ± "
                        f"{accuracy_std:.2f}"
                    ),

                "Total_MB":
                    (
                        f"{mb_mean:.2f} ± "
                        f"{mb_std:.2f}"
                    ),

                "Communication_Reduction_pct":
                    f"{reduction_mean:.2f}",

                "Accuracy_per_MB":
                    f"{acc_per_mb:.4f}",
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global fedavg_envelope_mb_global

    print(
        "=" * 100
    )

    print(
        "BATCS — E1 MAIN EXPERIMENT"
    )

    print(
        "=" * 100
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Clients: {NUM_CLIENTS}"
    )

    print(
        f"Clients/round: "
        f"{CLIENTS_PER_ROUND}"
    )

    print(
        "Participation: "
        f"{100.0 * CLIENTS_PER_ROUND / NUM_CLIENTS:.2f}%"
    )

    print(
        f"Rounds: {NUM_ROUNDS}"
    )

    print(
        f"Dirichlet alpha: "
        f"{DIRICHLET_ALPHA}"
    )

    print(
        f"Budget ratio: "
        f"{MAIN_BUDGET_RATIO:.2f}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    (
        train_set,
        train_eval_set,
        test_set,
    ) = load_cifar10()

    labels = np.asarray(
        train_set.targets,
        dtype=np.int64,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # ========================================================
    # SERIALIZED MODEL SIZE
    # ========================================================

    measurement_model = (
        CIFAR10CNN()
    )

    measurement_state = (
        clone_state_to_cpu(
            measurement_model.state_dict()
        )
    )

    model_mb = serialized_size_mb(
        measurement_state
    )

    (
        hard_budget_mb,
        fedavg_envelope_mb,
    ) = calculate_budget(
        model_mb
    )

    fedavg_envelope_mb_global = (
        fedavg_envelope_mb
    )

    print(
        f"Serialized model size: "
        f"{model_mb:.4f} MB"
    )

    print(
        f"FedAvg communication envelope: "
        f"{fedavg_envelope_mb:.2f} MB"
    )

    print(
        f"Common finite budget: "
        f"{hard_budget_mb:.2f} MB"
    )

    print(
        "Budget equation: "
        "B0 = full-horizon downlink + "
        "budget_ratio × full-uplink envelope"
    )

    print(
        "BATCS Stage-2: "
        "percentile evidence boundary + "
        "uncertainty-aware utility/byte + "
        "dynamic count + "
        "exact feasible subset + "
        "post-aggregation marginal feedback"
    )

    # ========================================================
    # RESULT COLLECTIONS
    # ========================================================

    all_summary = []
    all_rounds = []
    all_predictor = []

    # ========================================================
    # SEEDS
    # ========================================================

    for seed in SEEDS:

        print()
        print(
            "=" * 100
        )

        print(
            f"Creating partition for seed={seed}"
        )

        validation_indices, client_pool = (
            create_validation_split(
                labels,
                SERVER_VALIDATION_SIZE,
                seed,
            )
        )

        client_indices = (
            dirichlet_partition(
                labels,
                client_pool,
                NUM_CLIENTS,
                DIRICHLET_ALPHA,
                seed,
            )
        )

        sizes = np.asarray(
            [
                len(
                    client
                )
                for client
                in client_indices
            ],
            dtype=np.float64,
        )

        print(
            "Client samples "
            f"min={int(sizes.min())} "
            f"mean={sizes.mean():.1f} "
            f"max={int(sizes.max())}"
        )

        # ====================================================
        # METHODS
        # ====================================================

        for method in [
            "FedAvg",
            "Random+Budget",
            "UCB",
            "UCB+Budget",
            "BATCS",
        ]:

            print()
            print(
                "-" * 100
            )

            print(
                f"START {method} | seed={seed}"
            )

            method_budget = (
                hard_budget_mb
                if method
                in {
                    "Random+Budget",
                    "UCB+Budget",
                    "BATCS",
                }
                else math.inf
            )

            (
                summary,
                round_records,
                predictor_records,
            ) = run_method(
                method=method,
                seed=seed,
                train_set=train_set,
                train_eval_set=train_eval_set,
                test_loader=test_loader,
                client_indices=client_indices,
                validation_indices=validation_indices,
                model_mb=model_mb,
                budget_mb=method_budget,
            )

            all_summary.append(
                summary
            )

            all_rounds.extend(
                round_records
            )

            all_predictor.extend(
                predictor_records
            )

            print(
                f"DONE {method} | "
                f"seed={seed} | "
                f"accuracy={summary['final_accuracy']:.2f}% | "
                f"communication={summary['total_mb']:.2f} MB | "
                f"rounds={summary['rounds_completed']}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    summary_df = pd.DataFrame(
        all_summary
    )

    rounds_df = pd.DataFrame(
        all_rounds
    )

    predictor_df = pd.DataFrame(
        all_predictor
    )

    table_df = make_table(
        summary_df
    )

    summary_df.to_csv(
        RESULT_DIR
        /
        "E1_run_summary.csv",
        index=False,
    )

    rounds_df.to_csv(
        RESULT_DIR
        /
        "E1_round_results.csv",
        index=False,
    )

    predictor_df.to_csv(
        RESULT_DIR
        /
        "E1_BATCS_predictor.csv",
        index=False,
    )

    table_df.to_csv(
        RESULT_DIR
        /
        "E1_Table1.csv",
        index=False,
    )

    # ========================================================
    # PRIMARY BATCS vs UCB+Budget
    # ========================================================

    batcs_df = summary_df[
        summary_df[
            "method"
        ]
        ==
        "BATCS"
    ]

    ucb_budget_df = summary_df[
        summary_df[
            "method"
        ]
        ==
        "UCB+Budget"
    ]

    common_seeds = sorted(
        set(
            batcs_df[
                "seed"
            ]
        )
        &
        set(
            ucb_budget_df[
                "seed"
            ]
        )
    )

    if common_seeds:

        batcs_map = {
            int(row["seed"]):
                float(
                    row[
                        "final_accuracy"
                    ]
                )
            for _, row
            in batcs_df.iterrows()
        }

        ucb_map = {
            int(row["seed"]):
                float(
                    row[
                        "final_accuracy"
                    ]
                )
            for _, row
            in ucb_budget_df.iterrows()
        }

        paired_diffs = np.asarray(
            [
                batcs_map[
                    seed
                ]
                -
                ucb_map[
                    seed
                ]
                for seed
                in common_seeds
            ],
            dtype=np.float64,
        )

        comparison = {
            "method_a":
                "BATCS",

            "method_b":
                "UCB+Budget",

            "paired_runs":
                len(
                    common_seeds
                ),

            "accuracy_difference_mean_pp":
                float(
                    paired_diffs.mean()
                ),

            "communication_difference_mean_mb":
                float(
                    batcs_df[
                        "total_mb"
                    ].mean()
                    -
                    ucb_budget_df[
                        "total_mb"
                    ].mean()
                ),

            "note":
                (
                    "Descriptive comparison only; "
                    "BATCS is not forced to outperform the baseline, and "
                    "no claim of guaranteed superiority is made."
                ),
        }

    else:

        comparison = {
            "method_a":
                "BATCS",

            "method_b":
                "UCB+Budget",

            "paired_runs":
                0,

            "accuracy_difference_mean_pp":
                np.nan,

            "communication_difference_mean_mb":
                np.nan,
        }

    with open(
        RESULT_DIR
        /
        "E1_BATCS_vs_UCB_Budget.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            comparison,
            file,
            indent=2,
        )

    # ========================================================
    # REPRODUCIBILITY METADATA
    # ========================================================

    metadata = {
        "device":
            str(DEVICE),

        "num_clients":
            NUM_CLIENTS,

        "clients_per_round":
            CLIENTS_PER_ROUND,

        "participation_percentage":
            (
                100.0
                *
                CLIENTS_PER_ROUND
                /
                NUM_CLIENTS
            ),

        "num_rounds":
            NUM_ROUNDS,

        "local_epochs":
            LOCAL_EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "momentum":
            MOMENTUM,

        "dirichlet_alpha":
            DIRICHLET_ALPHA,

        "server_validation_size":
            SERVER_VALIDATION_SIZE,

        "seeds":
            SEEDS,

        "main_budget_ratio":
            MAIN_BUDGET_RATIO,

        "model_serialized_mb":
            model_mb,

        "fedavg_communication_envelope_mb":
            fedavg_envelope_mb,

        "common_finite_budget_mb":
            hard_budget_mb,

        "budget_definition":
            (
                "full-horizon downlink + "
                "MAIN_BUDGET_RATIO × full-horizon uplink"
            ),

        "communication_equation":
            (
                "B_used = "
                "B_downlink + "
                "B_preview + "
                "B_accepted_full"
            ),

        "rejected_full_updates_charged":
            False,

        "stage2_parameters":
            {
                "min_accept":
                    BATCS_MIN_ACCEPT,

                "max_accept":
                    BATCS_MAX_ACCEPT,

                "warmup_rounds":
                    BATCS_WARMUP_ROUNDS,

                "min_predictor_samples":
                    BATCS_MIN_PREDICTOR_SAMPLES,

                "soft_max_accept":
                    BATCS_SOFT_MAX_ACCEPT,

                "percentile":
                    BATCS_PERCENTILE,
                
                "borderline_pool_size":
                    BATCS_BORDERLINE_POOL,

                "uncertainty_beta":
                    BATCS_UNCERTAINTY_BETA,

                "max_predictor_weight":
                    BATCS_MAX_PREDICTOR_WEIGHT,

                "exploration_probability":
                    BATCS_EXPLORATION_PROB,

                "bad_trend_threshold_pp":
                    BATCS_BAD_TREND_THRESHOLD_PP,

                "low_score_separation":
                    BATCS_LOW_SCORE_SEPARATION,

                "high_score_separation":
                    BATCS_HIGH_SCORE_SEPARATION,
            },

        "final_stage2_patches":
            {
                "percentile_is_binding":
                    False,

                "percentile_is_evidence_boundary":
                    True,

                "borderline_pool_size":
                    BATCS_BORDERLINE_POOL,

                "dynamic_count_operates_over_expanded_affordable_pool":
                    True,

                "exact_subset_search_over_dynamic_candidate_pool":
                    True,

                "signed_risk_objective":
                    True,

                "ucb_protected_cold_start":
                    True,

                "conservative_predictor_weight":
                    True,

                "predictor_target_is_post_aggregation_marginal_validation_utility":
                    True,

                "current_round_target_leakage":
                    False,
            },
    }

    with open(
        RESULT_DIR
        /
        "E1_reproducibility.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        "TABLE 1 — E1 MAIN COMPARISON"
    )

    print(
        "=" * 100
    )

    print(
        table_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "BATCS vs UCB+Budget"
    )

    print(
        "=" * 100
    )

    print(
        json.dumps(
            comparison,
            indent=2,
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "E1 COMPLETED"
    )

    print(
        "=" * 100
    )

    print(
        f"Results saved in: "
        f"{RESULT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()