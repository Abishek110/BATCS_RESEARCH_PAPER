# BATCS --- Budget-Coupled Two-Stage Communication Control for Federated Learning

> **Research repository README --- E1--E5 frozen experimental pipeline**

BATCS is a communication-aware federated learning (FL) framework that
separates **client participation** from **full-update acquisition**.

The central idea is a two-stage communication-control mechanism:

``` text
50 total clients
      │
      ▼
Stage 1 — UCB client selection
      │
      │ exactly 10 clients selected before local training
      ▼
10 selected/training clients
      │
      ▼
Lightweight scalar previews
      │
      ▼
Stage 2 — preview + ridge prediction
         + uncertainty + UCB history
         + communication budget
      │
      │ typically 4–7 clients accepted
      ▼
Accepted full updates
      │
      ▼
FedAvg aggregation
```

The project is designed around an auditable finite communication budget.
It explicitly separates:

1.  **who trains** (Stage 1), from
2.  **whose complete update is actually acquired and aggregated** (Stage
    2).

The main scientific contribution is therefore **communication control
and budget-feasible update acquisition**, rather than a new CNN
architecture.

------------------------------------------------------------------------

## 1. Research idea

### Problem

In federated learning, communication can dominate the cost of
distributed training. A conventional FedAvg round typically downloads
the global model to participating clients and uploads complete client
updates from all selected clients.

However, not every client update is equally useful.

BATCS asks:

> **Can we first identify a small set of clients to train, obtain
> lightweight previews from them, and then acquire complete model
> updates only from the clients that appear most useful under a finite
> communication budget?**

The framework therefore treats communication as an explicit
resource-allocation problem.

### Core research hypothesis

A lightweight preview of a locally trained client update can contain
enough information to help the server decide which complete updates are
worth paying for.

The hypothesis is deliberately modest:

-   the preview predictor does **not** need to reconstruct every
    client's true contribution accurately;
-   it only needs to provide useful information for communication-aware
    ranking/selection;
-   the final decision must remain constrained by the actual serialized
    communication budget.

------------------------------------------------------------------------

# 2. What is novel about BATCS?

The strongest novelty claim is the **combination of two-stage selection
and explicit byte-budget control**.

### 2.1 Two-stage client/update control

BATCS distinguishes:

### Stage 1 --- participation selection

UCB selects exactly **10 of 50 clients** before local training.

This controls:

-   who performs local computation;
-   which clients are eligible to provide previews;
-   the first level of participation exploration/exploitation.

### Stage 2 --- full-update acquisition

The 10 selected clients train and generate lightweight previews.

The server then decides which clients should have their **complete model
updates acquired**.

The decision uses:

-   preview features;
-   an online ridge predictor;
-   predictor uncertainty;
-   UCB history;
-   utility per serialized communication byte;
-   an evidence percentile boundary;
-   controlled borderline exploration;
-   an exact budget-feasible subset optimizer.

Typically about **4--7 full updates** are accepted.

Thus the actual conceptual pipeline is:

``` text
50 clients → 10 selected → 4–7 accepted full updates
```

The frequently used example

``` text
100 → 50 → 25
```

is only an illustrative description of the architecture; it is **not**
the frozen BATCS experimental configuration.

------------------------------------------------------------------------

## 2.2 Budget-coupled communication

BATCS does not treat communication as an informal "number of clients"
constraint.

The frozen protocol explicitly tracks serialized byte usage.

The mandatory budget contains:

-   the full-horizon mandatory downlink;
-   a budget-ratio-dependent uplink envelope.

Actual communication additionally charges:

-   downlink;
-   lightweight previews;
-   accepted full updates.

**Rejected full updates are not charged**, because they are never
acquired.

The frozen main configuration uses:

``` text
MAIN_BUDGET_RATIO = 0.50
```

with an approximately 0.7235 MB serialized model.

------------------------------------------------------------------------

## 2.3 Preview-based utility prediction

The main FL model is a CNN.

The ridge regression model is **not** the federated model and does not
replace the CNN.

It is an auxiliary lightweight predictor used by Stage 2 to estimate the
expected usefulness of a client's complete update.

The conceptual roles are:

  -----------------------------------------------------------------------
  Component                           Role
  ----------------------------------- -----------------------------------
  CNN                                 Main federated learning model

  FedAvg                              Server aggregation

  UCB                                 Stage-1 client selection

  Ridge regression                    Stage-2 lightweight utility
                                      predictor

  Uncertainty estimate                Confidence/difficulty signal for
                                      the predictor

  Budget optimizer                    Chooses an affordable set of full
                                      updates
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 3. Frozen experimental protocol

Unless an experiment explicitly changes a parameter, the main BATCS
protocol is:

  Parameter                            Frozen setting
  -------------------------- ------------------------
  Dataset                                    CIFAR-10
  Total clients                                    50
  Stage-1 clients/round                    exactly 10
  Rounds                                           50
  Local epochs                                      1
  Batch size                                       64
  Learning rate                                  0.01
  Optimizer                                       SGD
  Momentum                                        0.9
  Dirichlet α                                     0.5
  Server validation size                        2,000
  Final seeds                  42, 123, 2024, 7, 2025
  Main budget ratio                              0.50
  Main serialized model                   \~0.7235 MB
  Full FedAvg envelope                    \~723.52 MB
  Common 50% finite budget                \~542.64 MB

### Methods

The frozen E1 comparison contains:

-   FedAvg
-   Random+Budget
-   UCB
-   UCB+Budget
-   BATCS

The controlled finite-budget baseline is **UCB+Budget**.

FedAvg and UCB are unconstrained references.

Random+Budget, UCB+Budget and BATCS operate under the finite-budget
protocol.

------------------------------------------------------------------------

# 4. Communication accounting

A critical reproducibility point is that BATCS uses **actual serialized
communication**, not an abstract client-count proxy.

Conceptually:

``` text
B_used =
    B_downlink
  + B_preview
  + B_accepted_full
```

where:

-   `B_downlink` = mandatory global-model downlink;
-   `B_preview` = serialized lightweight previews;
-   `B_accepted_full` = serialized complete updates for accepted
    clients.

Rejected complete updates are **not charged** because they are not
transmitted/acquired.

The serialized model size is measured from the actual implementation
rather than assumed from parameter count.

------------------------------------------------------------------------

# 5. Dataset: CIFAR-10

BATCS uses CIFAR-10.

The official CIFAR-10 source describes the dataset as **60,000 32×32
colour images in 10 classes**, with **50,000 training images and 10,000
test images**. The official download page provides the Python, Matlab
and binary versions. citeturn0search0

Official dataset page:

urlCIFAR-10 official dataset
pagehttps://www.cs.toronto.edu/\~kriz/cifar.html

### How CIFAR-10 is obtained

The original CIFAR-10 dataset is downloaded from the official CIFAR-10
distribution page.

For this project:

1.  Download the CIFAR-10 Python version.
2.  Extract it into the dataset location expected by the training code.
3.  The archive contains the standard:
    -   `data_batch_1`
    -   `data_batch_2`
    -   `data_batch_3`
    -   `data_batch_4`
    -   `data_batch_5`
    -   `test_batch`
    -   `batches.meta`
4.  The training split is partitioned into the federated clients using
    the experiment's non-IID partitioning procedure.
5.  The test set is reserved for final evaluation and is **not used as
    the predictor-training target**.

The official CIFAR-10 documentation states that the Python version
contains the five training batches and one test batch and describes the
serialized image/label layout. citeturn0search0

### Important reproducibility rule

Do **not** substitute a different dataset or silently change the
train/test split if you want to reproduce the frozen E1--E5 results.

The predictor must never use test-set accuracy as its target.

------------------------------------------------------------------------

# 6. Federated non-IID partitioning

The main experiments use a Dirichlet partition with:

``` text
alpha = 0.5
```

The E4 sensitivity experiment evaluates:

``` text
alpha ∈ {0.1, 0.5, 1.0}
```

Lower α produces a more heterogeneous/non-IID client distribution.

The experiment is intended to test whether the BATCS mechanism and
predictor remain usable under different levels of client heterogeneity.

------------------------------------------------------------------------

# 7. Repository structure

A recommended GitHub repository layout is:

``` text
BATCS/
│
├── README.md
│
├── src/
│   ├── E1.py
│   ├── E2.py
│   ├── E3.py
│   ├── E4.py
│   └── E5.py
│
├── requirements.txt
│
├── data/
│   └── cifar-10-batches-py/
│
├── batcs_results_final/
│   ├── E1_BATCS_predictor.csv
│   ├── ...
│   ├── E2_stage2_novelty_isolation/
│   ├── E3_budget_sensitivity/
│   ├── E4_noniid_sensitivity/
│   └── E5_preview_predictor_evaluation/
│
└── paper/
    ├── manuscript/
    ├── figures/
    └── tables/
```

Do not commit large raw datasets or unnecessarily large generated
artifacts to GitHub. Consider Git LFS or an external data-release
location for large files.

------------------------------------------------------------------------

# 8. Environment setup

The experiments were developed as Python programs and use PyTorch for
the federated CNN experiments.

Create a clean environment.

### macOS / Linux

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the dependencies listed in the project's `requirements.txt`.

For example:

``` bash
pip install -r requirements.txt
```

Verify Python:

``` bash
python --version
```

Verify PyTorch:

``` bash
python -c "import torch; print(torch.__version__)"
```

If using Apple Silicon/MPS, verify:

``` bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

The exact hardware/backend can affect runtime, so the repository should
record the environment used for the final published runs.

------------------------------------------------------------------------

# 9. Running the experiments

## Important: frozen E1--E4 artifacts

The original E1--E4 experiments are treated as **frozen scientific
artifacts**.

If the repository contains the already-generated final results, do not
rerun E1--E4 merely to reproduce E5.

E5 is a read-only analysis of the stored E1--E4 predictor records.

------------------------------------------------------------------------

# 10. E1 --- Main BATCS experiment

## Purpose

E1 is the main end-to-end experiment.

It compares:

``` text
FedAvg
Random+Budget
UCB
UCB+Budget
BATCS
```

under the primary CIFAR-10, α=0.5, 50% budget configuration.

## What E1 does

Each run:

1.  Initializes the federated CNN.
2.  Creates the 50-client non-IID CIFAR-10 federation.
3.  Runs 50 communication rounds.
4.  Stage 1 selects exactly 10 clients using UCB for BATCS/UCB.
5.  Selected clients perform one local epoch.
6.  BATCS clients generate lightweight previews.
7.  Stage 2 predicts utility and allocates full-update communication.
8.  The exact budget-feasible subset is selected.
9.  Accepted full updates are aggregated with FedAvg.
10. Server validation is performed.
11. Final test accuracy is reported.

### Run

From the directory containing `E1.py`:

``` bash
python E1.py
```

The script uses its configured dataset/result paths. If running on a new
machine, verify the dataset root and output root in the configuration
section before starting.

## Main frozen E1 results

  Method            Test accuracy (%)   Communication (MB)
  --------------- ------------------- --------------------
  FedAvg                 42.51 ± 2.64               723.52
  Random+Budget          36.47 ± 2.78               535.40
  UCB                    43.32 ± 2.16               723.52
  UCB+Budget             41.11 ± 3.34               535.40
  BATCS                  41.83 ± 4.67               542.08

BATCS versus the controlled UCB+Budget baseline:

``` text
Accuracy difference: +0.716 percentage points
Communication difference: +6.679 MB
```

This is a descriptive result. It should **not** be presented as a
statistically significant superiority claim.

------------------------------------------------------------------------

# 11. E2 --- Stage-2 novelty isolation

## Purpose

E2 asks:

> Does the learned Stage-2 predictor contribute beyond the underlying
> selection/budget machinery?

It compares:

``` text
UCB+Budget
BATCS-NoPredictor
BATCS
```

The Stage-1 mechanism and communication-control machinery remain
aligned.

### BATCS-NoPredictor

The NoPredictor variant removes the learned ridge predictor and learned
uncertainty.

It uses the pre-decision UCB score as a non-learned proxy and does not
use current-round post-decision information to make the decision.

This isolates the contribution of the learned Stage-2 predictor.

### Run

``` bash
python E2.py
```

## Frozen E2 results

  Method                Accuracy (%)   Communication (MB)
  ------------------- -------------- --------------------
  UCB+Budget            41.11 ± 3.34               535.40
  BATCS-NoPredictor     40.37 ± 3.23               542.08
  BATCS                 41.83 ± 4.67               542.08

BATCS minus BATCS-NoPredictor:

``` text
+1.46 percentage points
```

Across the five seeds, BATCS wins 4/5 seeds.

No statistical-significance claim is made from these five paired runs.

### Interpretation

E2 is a **novelty isolation/ablation experiment**, not proof that the
predictor always improves accuracy.

Its purpose is to show that the learned Stage-2 mechanism contributes
beyond a non-predictive Stage-2 alternative.

------------------------------------------------------------------------

# 12. E3 --- Budget sensitivity

## Purpose

E3 evaluates how BATCS behaves as the finite communication budget
changes.

Budget ratios:

``` text
25%
50%
75%
100%
```

Methods:

``` text
Random+Budget
UCB+Budget
BATCS
```

Seeds:

``` text
42, 123, 2024, 7, 2025
```

### Run

``` bash
python E3.py
```

## Frozen E3 summary

    Budget Method            Accuracy (%)   Communication (MB)   Rounds
  -------- --------------- -------------- -------------------- --------
       25% Random+Budget     34.51 ± 1.80               448.58       31
       25% UCB+Budget        37.79 ± 2.95               448.58       31
       25% BATCS             28.71 ± 6.47               451.64       50
       50% Random+Budget     36.47 ± 2.78               535.40       37
       50% UCB+Budget        41.11 ± 3.34               535.40       37
       50% BATCS             41.83 ± 4.67               542.08       50
       75% Random+Budget     38.81 ± 3.31               622.22       43
       75% UCB+Budget        43.02 ± 2.62               622.22       43
       75% BATCS             41.54 ± 5.08        543.09 ± 0.40       50
      100% Random+Budget     42.51 ± 2.64               723.52       50
      100% UCB+Budget        43.32 ± 2.16               723.52       50
      100% BATCS             41.54 ± 5.08        543.09 ± 0.40       50

### Main E3 observation

The strongest E3 result is not that BATCS always has the highest
accuracy.

It is:

> **BATCS can complete the full 50-round horizon while using
> substantially less communication than the unconstrained/full-budget
> references at higher budget ratios.**

At tight budgets, UCB+Budget exhausts its available communication
earlier, while BATCS continues through the planned 50 rounds.

Therefore E3 supports a **communication-efficiency / round-completion
tradeoff**, not a universal accuracy-superiority claim.

------------------------------------------------------------------------

# 13. E4 --- Non-IID sensitivity

## Purpose

E4 tests BATCS under different Dirichlet heterogeneity levels.

``` text
alpha = 1.0
alpha = 0.5
alpha = 0.1
```

Methods:

``` text
UCB+Budget
BATCS
```

Budget:

``` text
50%
```

Seeds:

``` text
42, 123, 2024, 7, 2025
```

Only α is changed.

### Run

``` bash
python E4.py
```

## Frozen E4 results

      α Method         Accuracy (%)   Communication (MB)   Rounds
  ----- ------------ -------------- -------------------- --------
    1.0 UCB+Budget     42.49 ± 2.49               535.40       37
    1.0 BATCS          41.44 ± 2.37               542.08       50
    0.5 UCB+Budget     41.11 ± 3.34               535.40       37
    0.5 BATCS          41.83 ± 4.67               542.08       50
    0.1 UCB+Budget     25.97 ± 6.65               535.40       37
    0.1 BATCS          22.77 ± 8.83               542.08       50

### E4 interpretation

As non-IID heterogeneity becomes stronger, overall accuracy decreases
for both methods.

BATCS still completes the full 50-round horizon under the finite
communication budget.

The accuracy results do not demonstrate universal BATCS superiority. The
correct conclusion is that the communication-control mechanism remains
operational under the tested heterogeneity regimes, while accuracy
remains sensitive to the underlying FL difficulty.

### Reproducibility note

The final E4 launcher completed all 30 jobs. One BATCS α=0.1, seed=42
attempt encountered a genuine segmentation fault and succeeded on a
fresh retry. This should be retained in the experiment log rather than
silently removed from the history.

------------------------------------------------------------------------

# 14. E5 --- Predictor evaluation

## Purpose

E5 is a **read-only post-hoc analysis**.

It does not retrain FL.

It does not modify the predictor.

It does not fit a replacement predictor.

It evaluates the stored online predictions generated by E1--E4.

The E5 protocol uses:

``` text
Primary target:
Observed_Marginal_Utility_pp

Secondary target:
Isolated_Utility_pp

Prediction:
Predicted_Utility_pp

Uncertainty:
Uncertainty_pp
```

The E5 manifest records:

``` text
Protocol version: 2.0
Combined rows: 22,500
Jobs: 45
FL training: false
Torch: false
MPS: false
Test accuracy as predictor target: false
```

The stored E5 analysis explicitly treats the primary target as censored:
rejected clients do not receive an imputed primary label. Isolated
utility is evaluated separately as a fully observed secondary
diagnostic. fileciteturn258file0L2-L13
fileciteturn258file0L137-L145

### Run

From the directory containing `E5.py`:

``` bash
python E5.py
```

E5 reads the frozen E1--E4 predictor CSVs and writes analysis outputs.

### E5 primary result

For E1:

``` text
MAE      = 1.597509 pp
RMSE     = 2.198248 pp
R²       = 0.067015
Spearman = 0.286485
```

This indicates **modest predictive signal**, not a highly accurate
predictor.

That is an important result.

### E5 by non-IID condition

      α   MAE (pp)       R²   Spearman
  ----- ---------- -------- ----------
    0.1      1.173    0.036      0.171
    0.5      1.598    0.067      0.286
    1.0      1.432   -0.037      0.112

### E5 by budget

    Budget   MAE (pp)       R²   Spearman
  -------- ---------- -------- ----------
       25%      1.552   -0.008      0.122
       50%      1.598    0.067      0.286
       75%      1.598    0.061      0.274
      100%      1.598    0.061      0.274

The predictor does not materially improve when the communication budget
increases beyond 50% in these experiments.

------------------------------------------------------------------------

# 15. E5 ranking and decision diagnostics

E5 additionally evaluates:

-   top-1 regret;
-   top-3 regret;
-   top-5 regret;
-   predictor-vs-baseline ranking performance;
-   accepted-vs-rejected isolated utility;
-   positive-utility precision/recall;
-   uncertainty/error behavior by decile.

### Important censoring rule

The primary marginal-utility target is available only for accepted
clients.

Therefore:

``` text
Rejected client
      ↓
No observed primary marginal label
      ↓
Do NOT replace with zero
```

E5 never imputes rejected primary targets.

This avoids a serious evaluation bias.

The fully observed `Isolated_Utility_pp` field is used only as a
**secondary counterfactual diagnostic**, not as a replacement for the
online training target. fileciteturn258file0L137-L145

------------------------------------------------------------------------

# 16. E5 uncertainty findings

The uncertainty diagnostic shows that the stored uncertainty signal is
useful as a difficulty indicator, but it is **not perfectly calibrated
as a probabilistic confidence interval**.

Therefore the paper should not claim:

> "BATCS has calibrated uncertainty."

A more accurate statement is:

> "The uncertainty estimate provides a diagnostic signal associated with
> prediction difficulty, although empirical coverage indicates imperfect
> calibration."

------------------------------------------------------------------------

# 17. What each experiment contributes to the paper

  --------------------------------------------------------------------------
  Experiment              Scientific question     Main contribution
  ----------------------- ----------------------- --------------------------
  E1                      Does BATCS work         Main BATCS vs baseline
                          end-to-end?             comparison

  E2                      Does Stage-2 prediction Novelty isolation /
                          matter?                 predictor ablation

  E3                      What happens under      Communication-efficiency
                          different budgets?      and round-completion
                                                  analysis

  E4                      Does it survive         Robustness/sensitivity
                          different non-IID       
                          conditions?             

  E5                      Does the predictor      Predictor, ranking,
                          actually contain useful decision and uncertainty
                          signal?                 analysis
  --------------------------------------------------------------------------

Together:

``` text
E1 → End-to-end effectiveness
E2 → Mechanism isolation
E3 → Budget behavior
E4 → Non-IID robustness
E5 → Predictor validity
```

------------------------------------------------------------------------

# 18. What BATCS does NOT claim

The following claims should be avoided:

-   BATCS always has the highest accuracy.
-   BATCS statistically significantly beats UCB+Budget from only five
    seeds.
-   The ridge predictor is highly accurate.
-   The uncertainty estimate is perfectly calibrated.
-   The predictor accurately predicts every client's true contribution.
-   BATCS is universally better on every dataset.
-   The current experiments prove real-world network/energy superiority.
-   Rejected-client marginal utility is known.
-   Test accuracy is used to train the predictor.

The strongest defensible contribution is:

> **BATCS provides a two-stage, budget-coupled mechanism for controlling
> which client updates are fully acquired in federated learning, with
> explicit serialized-byte accounting and an online lightweight
> predictor used to inform Stage-2 selection.**

------------------------------------------------------------------------

# 19. Main scientific limitations

The current study should explicitly acknowledge:

### Dataset scope

The frozen study uses CIFAR-10.

Therefore external validity to:

-   FEMNIST;
-   CIFAR-100;
-   medical datasets;
-   large-scale production FL;

is not established by E1--E5.

### Number of seeds

The principal comparisons use five seeds.

This provides paired replication but is not sufficient for broad claims
of statistical significance.

### Systems realism

The experiments are synchronous simulations.

They do not fully model:

-   client availability;
-   network latency;
-   packet loss;
-   stragglers;
-   secure aggregation overhead;
-   device energy;
-   heterogeneous bandwidth;
-   privacy mechanisms.

### Predictor limitations

The predictor has modest R²/Spearman performance and imperfect
uncertainty calibration.

This should be presented as a limitation rather than hidden.

------------------------------------------------------------------------

# 20. Reproducibility and artifact integrity

The frozen E1 SHA-256 used throughout the later experiments is:

``` text
40249a6d3407865692eca22841a7940445642c9661322ab3cbd3f4232a578410
```

E2, E3 and E4 validate this frozen E1 source hash in their source
metadata.

E5 also records this expected frozen E1 SHA.
fileciteturn258file0L13-L13

E5 source coverage:

``` text
E1:  2,500 predictor rows
E2:  2,500 predictor rows
E3: 10,000 predictor rows
E4:  7,500 predictor rows
--------------------------
Total: 22,500 rows
```

E5 evaluates:

``` text
45 jobs
```

The E5 manifest also records the expected five seeds, E3's four budget
ratios and E4's three α values. fileciteturn258file0L54-L55
fileciteturn258file0L65-L121

------------------------------------------------------------------------

# 21. Recommended execution order

For a **fresh full reproduction**:

``` bash
python E1.py
python E2.py
python E3.py
python E4.py
python E5.py
```

For **analysis-only reproduction using the frozen E1--E4 artifacts**:

``` bash
python E5.py
```

This is the recommended route when the final E1--E4 CSV artifacts are
already available.

------------------------------------------------------------------------

# 22. Running from a GitHub clone

Example:

``` bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd BATCS
```

Create the environment:

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place CIFAR-10 in the expected dataset location.

Then:

``` bash
cd src
python E1.py
python E2.py
python E3.py
python E4.py
python E5.py
```

If the scripts use absolute paths in the frozen local version, update
the configured dataset/output roots for the new machine before
execution. Do not change the scientific hyperparameters if the goal is
exact reproduction.

------------------------------------------------------------------------

# 23. Output artifacts

The project produces experiment-specific result directories.

Typical outputs include:

``` text
E1_BATCS_predictor.csv

E2_stage2_novelty_isolation/
    E2_BATCS_predictor.csv

E3_budget_sensitivity/
    E3_BATCS_predictor.csv

E4_noniid_sensitivity/
    E4_BATCS_predictor.csv

E5_preview_predictor_evaluation/
    E5_summary.csv
    E5_primary_main.csv
    E5_by_alpha.csv
    E5_by_budget.csv
    E5_job_metrics.csv
    E5_round_metrics.csv
    E5_ranking_quality.csv
    E5_decision_quality.csv
    E5_uncertainty_calibration.csv
    E5_false_rejection.csv
    E5_all_predictor_records.csv
    E5_manifest.json
    plots/
```

The E5 manifest records the five generated plot paths and the counts of
ranking, decision and uncertainty-calibration rows.
fileciteturn258file0L126-L135

------------------------------------------------------------------------

# 24. Paper positioning

## Suggested paper title

### Primary option

**BATCS: Budget-Coupled Two-Stage Communication Control for Federated
Learning Using Multi-Armed Bandits**

Alternative:

**Budget-Coupled Two-Stage Client Selection and Update Acquisition for
Communication-Efficient Federated Learning**

The first title is recommended because it directly names the framework.

------------------------------------------------------------------------

# 25. Suggested paper contribution statements

The paper can be organized around four main contributions.

### Contribution 1 --- Two-stage communication control

We introduce a two-stage FL communication-control framework that
separates pre-training client participation from post-training
full-update acquisition.

### Contribution 2 --- Budget-coupled update acquisition

We formulate Stage-2 update acquisition as an explicit finite-byte
allocation problem rather than relying solely on a fixed number of
selected clients.

### Contribution 3 --- Preview-based utility estimation

We use lightweight client previews and an online ridge predictor with
uncertainty to estimate the usefulness of complete updates before
incurring their full communication cost.

### Contribution 4 --- Auditable evaluation

We evaluate the method using explicit serialized communication
accounting, controlled baselines, budget sensitivity, non-IID
sensitivity, predictor ablation, and post-hoc predictor diagnostics.

------------------------------------------------------------------------

# 26. Recommended paper structure

## I. Introduction

Explain:

1.  FL communication is expensive.
2.  Conventional participation selection decides who trains, but not
    necessarily whose complete update is worth acquiring.
3.  Communication should be treated as a finite resource.
4.  BATCS separates participation from full-update acquisition.
5.  State contributions.

## II. Related Work

Suggested themes:

-   federated client selection;
-   multi-armed bandits for FL;
-   communication-efficient FL;
-   resource-aware FL;
-   gradient/update compression;
-   client utility estimation;
-   adaptive aggregation.

## III. System Model

Define:

-   clients;
-   global model;
-   rounds;
-   local training;
-   serialized communication;
-   budget;
-   server validation.

## IV. BATCS Method

### IV-A. Stage 1 UCB selection

Exactly 10 clients selected from 50.

### IV-B. Lightweight preview

Define preview features.

### IV-C. Online ridge predictor

Explain:

-   feature vector;
-   ridge update;
-   prediction;
-   uncertainty;
-   prequential nature.

### IV-D. Stage-2 acceptance

Explain:

-   utility;
-   utility per byte;
-   evidence percentile;
-   borderline exploration;
-   dynamic acceptance count;
-   exact subset optimization.

### IV-E. Communication accounting

Give the exact byte equation.

## V. Experimental Setup

Describe:

-   CIFAR-10;
-   50 clients;
-   Dirichlet α=0.5;
-   50 rounds;
-   local epoch;
-   batch size;
-   optimizer;
-   five seeds;
-   validation split;
-   serialized communication.

## VI. Results

### VI-A. E1

Main comparison.

### VI-B. E2

Stage-2 novelty isolation.

### VI-C. E3

Budget sensitivity.

### VI-D. E4

Non-IID sensitivity.

### VI-E. E5

Predictor quality and decision diagnostics.

## VII. Limitations

Explicitly discuss:

-   CIFAR-10-only scope;
-   five seeds;
-   simulated synchronous communication;
-   no real network latency/energy;
-   imperfect predictor calibration.

## VIII. Conclusion

Return to the central claim:

> BATCS turns client-update acquisition into an explicit
> communication-budget allocation problem and separates participation
> from full-update acquisition.

------------------------------------------------------------------------

# 27. Recommended interpretation of the overall results

The overall results should be interpreted as a
**systems/communication-control contribution**, not as a claim that
BATCS universally improves classification accuracy.

The strongest evidence is:

1.  BATCS can complete the planned 50 rounds under budgets where
    UCB+Budget exhausts its budget earlier.
2.  BATCS performs Stage-2 acquisition using actual serialized
    communication.
3.  E2 shows that the learned Stage-2 mechanism contributes beyond the
    NoPredictor variant.
4.  E3 demonstrates meaningful budget-dependent behavior.
5.  E4 shows operation across multiple non-IID regimes.
6.  E5 confirms that the predictor contains modest predictive/ranking
    signal while honestly exposing its limitations.

This combination is more important than any single accuracy number.

------------------------------------------------------------------------

# 28. Reproducibility checklist

Before releasing the GitHub repository, verify:

``` text
[ ] README.md included
[ ] requirements.txt included
[ ] E1.py included
[ ] E2.py included
[ ] E3.py included
[ ] E4.py included
[ ] E5.py included
[ ] dataset instructions included
[ ] random seeds documented
[ ] budget ratios documented
[ ] model architecture documented
[ ] optimizer documented
[ ] communication accounting documented
[ ] frozen E1 SHA documented
[ ] E1–E4 result CSVs preserved
[ ] E5 manifest preserved
[ ] E5 plots preserved
[ ] paper figures separated from raw experiment outputs
[ ] large dataset files excluded from normal Git history
[ ] no private absolute filesystem paths in the public release
[ ] no API keys/secrets included
```

Before publication, replace any machine-specific absolute paths such as:

``` text
/Volumes/Abishek Personal/Downloads/...
```

with repository-relative paths or configuration variables.

------------------------------------------------------------------------

# 29. Short project description for GitHub

**BATCS (Budget-Coupled Two-Stage Communication Control for Federated
Learning)** is a communication-aware federated learning framework that
separates client participation from full-update acquisition. Stage 1
uses multi-armed-bandit client selection, while Stage 2 uses lightweight
client previews, online utility prediction, uncertainty, and an exact
serialized-byte budget to decide which complete client updates should be
acquired and aggregated. The repository contains five frozen experiments
covering end-to-end performance, Stage-2 ablation, budget sensitivity,
non-IID sensitivity, and post-hoc predictor evaluation.

------------------------------------------------------------------------

# 30. Citation / dataset acknowledgement

If using CIFAR-10, cite the original CIFAR-10 technical report
associated with the official dataset distribution.

The official dataset page identifies:

**Alex Krizhevsky, 2009, *Learning Multiple Layers of Features from Tiny
Images*.**

The official page also asks users of the dataset to cite the technical
report. citeturn0search0

Official source:

urlCIFAR-10 / CIFAR-100 official distribution
pagehttps://www.cs.toronto.edu/\~kriz/cifar.html

------------------------------------------------------------------------

# 31. Final scientific status

### E1 --- FROZEN

Main end-to-end BATCS evaluation.

### E2 --- FROZEN

Stage-2 novelty isolation.

### E3 --- FROZEN

Budget sensitivity.

### E4 --- FROZEN

Non-IID sensitivity.

### E5 --- FROZEN

Read-only predictor evaluation.

The E5 artifact is explicitly marked as read-only post-hoc analysis and
records that it does not use FL training, Torch/MPS, or test accuracy as
the predictor target. fileciteturn258file0L2-L10

**Do not modify the frozen experimental results to improve numerical
appearance.**

The scientific value of the project comes from the complete, auditable
evaluation pipeline and from honestly reporting both strengths and
limitations.

------------------------------------------------------------------------

# 32. One-sentence summary

> **BATCS is a two-stage, budget-coupled federated learning
> communication-control framework that first selects clients for local
> training and then selectively acquires their complete updates using
> lightweight previews, utility prediction, uncertainty, and exact
> serialized-byte budgeting.**
