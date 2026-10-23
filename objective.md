# Autonomous TCR–pMHC model research — initial objective

## Task

You are an autonomous ML research agent. Your goal is to improve prediction of whether a T-cell receptor (TCR) binds a peptide–MHC (pMHC) complex, using **amino-acid sequence inputs only**.

**`ml.py` is the only existing source file you may modify.** It contains the initial, intentionally simple model and training implementation. Treat it as the starting point, not an architectural constraint. You have full freedom to replace or redesign the model architecture, feature representation and optimisation approach **within `ml.py`**, including MLPs, CNNs, transformers, attention, protein language model encoders or representation-learning objectives, subject to the available compute and libraries. Do not alter the underlying dataset, splits, evaluation definitions or experimental runner.

For this first pass, **validation mean squared error (MSE) is the primary score to minimise**. Labels are binary (`binding_flag`: 1 = recorded binding-positive pair, 0 = negative/decoy pair), but the starter model is scored using MSE between the label and a predicted binding score. Report AUROC as a secondary, complementary diagnostic where possible. Do not silently replace the fixed primary metric with another metric. If predicting probabilities, outputs must be in [0, 1] for MSE. You may use a different differentiable training loss if useful, but selection remains on the same held-out validation MSE.

## Data provided

Three CSVs are provided at repository-configured paths (inspect the actual local paths; do not assume a directory). Do not edit them.

The CSV paths will be defined under `TRAIN_PATH`, `VALIDATION_PATH` and `TEST_PATH`

| Split | Filename | Rows | Positive (`1`) | Negative (`0`) |
| --- | --- | ---: | ---: | ---: |
| Train | `train_multiview_negative_decoys_hla_v2_immrep_relabelled.csv` | 56,120 | 27,516 | 28,604 |
| Validation | `val_multiview_hla_v2.csv` | 3,888 | 1,944 | 1,944 |
| Test | `test_multiview_hla_v2.csv` | 4,364 | 2,182 | 2,182 |
| **Total** | | **64,372** | **31,642** | **32,730** |

The files contain TCR sequence, peptide sequence, HLA sequence, binding labels and ancillary metadata. The relevant input and target columns are:

- `TCR_full`: concatenated TCR α- and β-chain amino-acid sequences; use `tcra_len` and `tcrb_len` to separate them at the correct boundary.
- `Peptide`: presented peptide amino-acid sequence.
- `HLA_sequence`: HLA amino-acid sequence. Keep MHC representation consistent across all splits.
- `binding_flag`: binary target (1 positive, 0 negative).

**Full-chain restriction:** train and evaluate only rows with *both* α and β chains present (`tcra_len > 0` and `tcrb_len > 0`), using the complete chain sequences, not CDR3-only fragments. The CSVs have the following eligible row counts, before any additional filtering:

| Split | Both α and β present | Excluded for missing α or β |
| --- | ---: | ---: |
| Train | 48,912 | 7,208 |
| Validation | 3,342 | 546 |
| Test | 3,800 | 564 |

Verify that `tcra_len + tcrb_len == len(TCR_full)` and that relevant inputs/labels are valid. The reported counts are for initial orientation; the runner should compute actual eligible class counts after filtering. Do not infer α/β boundaries from sequence motifs or train on single-chain examples. The HLA field is a sequence representation, not a guarantee of fully observed MHC class-I multichain biology.

The training file includes *constructed decoy negatives*, and 544 labels are marked as relabelled based on IMMREP experimental evidence (`immrep_label_action`). Negative labels must therefore not be described as uniformly experimentally confirmed non-binding events. Use the supplied `binding_flag` without silently changing labels.

**Data provenance.** The underlying TCR–epitope research draws on the following published resources; these citations do not imply that every final row can be individually attributed to one paper without a row-level provenance audit:

1. Vita R, et al. *The Immune Epitope Database (IEDB): 2024 update.* Nucleic Acids Research. 2025;53(D1):D436–D443. https://doi.org/10.1093/nar/gkae1092
2. Goncharov M, Bagaev D, Shcherbinin D, et al. *VDJdb in the pandemic era: a compendium of T cell receptors specific for SARS-CoV-2.* Nature Methods. 2022;19:1017–1019. https://doi.org/10.1038/s41592-022-01578-0
3. Mayer-Blackwell K, Schattgen S, Cohen-Lavi L, et al. *TCR meta-clonotypes for biomarker discovery with tcrdist3 enabled identification of public, HLA-restricted clusters of SARS-CoV-2 TCRs.* eLife. 2021;10:e68605. https://doi.org/10.7554/eLife.68605

## Autonomous experiment loop

1. Inspect `ml.py` and the data schema. Run the initial model once to establish a baseline using the prescribed validation split and full-chain filter.
2. Propose **one** meaningful architectural or training change and state the hypothesis in a short sentence.
3. Modify **only `ml.py`**, run the fixed training/evaluation command, and collect measured results.
4. Compare validation MSE against the best valid previous run. Keep a change only if it improves the predetermined validation score; otherwise restore the previous best version of `ml.py`.
5. Repeat autonomously within the configured experiment and compute budget. You choose what to explore next; there is **no predefined architecture search space**.

**Do not inspect test labels, tune against the test set or change validation membership.** Test evaluation, if authorised, happens after selecting and freezing the final model. Do not use `boltz_embedding_npz`, structural files, AF3 outputs, metadata suggesting labels or split provenance as model features. Do not download or train very large encoders without checking available storage, GPU budget and external-data rules.

## Logging and progress graph

The orchestration code (outside `ml.py`) should own the logs and evaluation protocol; it is **not** part of the agent's editable surface. If logging infrastructure does not already exist, ask the developer/operator to provision it before autonomous experimentation; do not change other existing source files yourself.

For every attempted experiment, record: experiment number, timestamp, hypothesis, description of the `ml.py` change, revision/hash, validation MSE, validation AUROC (if computable), runtime, status (kept/discarded/failed), and the best-so-far validation MSE. Persist this as an append-only file such as `experiments/results.jsonl`, plus human-readable `experiments/summary.md`; these are **generated outputs, not editable source files**. Feed the prior experiment summary and relevant results back into the Argon context before the next decision.

Produce a progress plot, saved as `experiments/progress.png`, modelled on the autoresearch example: experiment number on the x-axis, **validation MSE (lower is better)** on the y-axis, grey points for discarded experiments, green points for accepted improvements, and a green step line for best-so-far MSE. Do not call this bits-per-byte: MSE on binary binding labels is not language-model BPB. Log secondary AUROC separately.

## Completion

Demonstrate at least one full autonomous cycle (proposal → edit `ml.py` → train/evaluate → log → next decision). Report the baseline, all experiments, best validation MSE, secondary AUROC, and whether the best model improved. Accurate, reproducible negative results are acceptable. Above all, **do not modify any existing file except `ml.py`**.
