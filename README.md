# Autonomous Research for TCR–pMHC Binding

# TCR-pMHC Binding - The Mark of Immunity

The immune system uses T-cell receptors (TCRs) to recognise potentially harmful cells, such as viruses and cacner. Cells display small fragments of proteins, called peptides, on their surface using molecules known as MHCs, forming peptdie-MHC (pMHC) complexes. When a TCR binds to a specific pMHC complex, it signals that the T cell has recognised its target and can trigger an immune response. Predicting which TCRs bind to which pMHC complexes is therefore a fundamental step towards understanding and harnessing the immune system.

![TCR–pMHC binding cartoon](./figures/tcr_pmhc_cartoon.png)

## Why this matters

If we could reliably predict which TCRs bind to which antigens, it would have major implications for **cancer immunotherapy**, **autoimmune disease**, **infectious disease monitoring**, and **personalised vaccine design**.

Potential applications include:

- **Personalised cancer vaccines**: prioritising peptides most likely to be immunogenic.
- **Disease monitoring**: identifying TCR signatures associated with infection, tumour response, or immune dysfunction.
- **Autoimmune research**: understanding unwanted TCR recognition of self-like peptides.
- **Therapeutic discovery**: accelerating peptide-specific TCR discovery and screening.

## Why it is hard - and why we need ML

The scale and complexity of TCR–pMHC recognition makes exhaustive experimental testing impossible. In recent years, researchers have therefore shifted towards developing predictive ML models.

Enormous diversity: over 10¹⁵ theoretically possible TCRs, with only a fraction experimentally characterised.

Complex binding: hypervariable CDR loops within the TCR and flexible docking geometries make molecular recognition difficult to predict.

Immune escape: even single amino-acid mutations can disrupt recognition, allowing viruses and tumours to evade immunity.

Sparse, biased data: limited experimental observations make it difficult for models to generalise to unseen interactions.

The challenge is to learn transferable rules of molecular recognition, rather than memorise known binding pairs.

## Our thesis

We are testing whether **Gemini 4 Argon** can act as an autonomous research scientist for this problem.

The agent is given:
- a fixed dataset of TCR–pMHC sequence pairs,
- an initial modelling file (`ml.py`),
- a fixed evaluation pipeline (`eval.py`),
- and an experiment log capturing prior hypotheses, edits, results, and decisions.

Argon then iteratively:
1. proposes a scientifically motivated modelling change,
2. edits **only** `ml.py`,
3. runs training and evaluation,
4. reads the resulting metrics and updated experiment log,
5. updates its reasoning using the **large context window**, and
6. decides what to try next.

The focus is contextual scientific iteration, not only model optiminsation: each experiment is informed by the growing research history, allowing the agent to build hypotheses cumulatively rather than treating each run in isolation.

## Research goal

Our immediate aim is to improve **validation performance under scientifically defensible evaluation**, while generating a transparent record of:
- what was changed,
- why it was changed,
- what happened,
- and whether the evidence supports keeping the modification.

More broadly, this project asks a simple but important question:

> Can a frontier model use long-context experimental memory to conduct meaningful, iterative machine learning research on a hard biological problem?

## Data provenance

This work draws on curated TCR–epitope resources including:
- **IEDB**: Vita et al., *Nucleic Acids Research* (2025)
- **VDJdb**: Goncharov et al., *Nature Methods* (2022)
- **tcrdist3-related public SARS-CoV-2 TCR analyses**: Mayer-Blackwell et al., *eLife* (2021)

## Repository structure

- `ml.py` — the **only** model file the agent may modify
- `eval.py` — fixed evaluation logic for model and agent assessment
- `objective.md` — the agent’s research brief
- `experiments/` — logs, results, summaries, and progress plots

## How to run

Each launch creates `experiments/<session>/` with one subdirectory per experiment. `--resume` reopens the latest session. `--session NAME` reopens a named one. The repository `ml.py` is never written.

Set `GEMINI_API_KEY` or `OPENROUTER_API_KEY`, plus `TRAIN_PATH` and `VALIDATION_PATH`, in `.env`. `TEST_PATH` is only for final evaluation.

```bash
python agent/agent.py --dry-run
python agent/agent.py --max-experiments 2 --run-timeout 900
python agent/agent.py --provider openrouter --model vendor/name --max-experiments 2 --max-input-tokens 100000 --max-output-tokens 8192
python agent/agent.py --resume --max-experiments 2
python agent/agent.py --request-stop
```

Gemini is the default (`gemini-4-argon`). The default budget is 900 seconds per training or evaluation process and one hour per session. A second controller cannot start while `experiments/.lock` is held.

Ctrl+C or `--request-stop` is checked between experiments and during training. The child gets SIGINT and about ten seconds to save `last_model.pth` before it is killed. `model.pth` is the checkpoint `eval.py` scores.

```bash
python final_eval.py --test-path /absolute/path/to/test.csv --session SESSION --experiment-id 0001
```

Final evaluation freezes that session. Later `--resume` on it is refused, so the test labels cannot steer another parent choice.

## Bottom line

TCR–pMHC binding prediction sits at the intersection of **machine learning, structural biology, and immunology**, with direct relevance to cancer, autoimmunity, infection, and precision therapeutics.

This repository explores whether autonomous long-context reasoning can move that problem forward.
