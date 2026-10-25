# TCR–pMHC binding

Improve prediction of whether a T-cell receptor binds a peptide–MHC complex, using amino-acid sequences only.

The repository `ml.py` is the starter and is never overwritten. Each experiment names a parent (`starter` or an earlier experiment id) and states a falsifiable hypothesis. A hypothesis may be an architecture change, a small precise adjustment, or a different representation or training setup if the current line has stalled. Do not combine independent hypotheses in one experiment.

## Data

Use only `TCR_full`, `Peptide`, and `HLA_sequence`. `binding_flag` is the label. Train on rows with both alpha and beta chains (`tcra_len > 0` and `tcrb_len > 0`) and check `tcra_len + tcrb_len == len(TCR_full)`. Do not infer chain boundaries from motifs, and do not train on single-chain rows.

Do not use structural features, AF3 outputs, `boltz_embedding_npz`, category annotations, provenance metadata, or label-derived fields as model inputs. Do not train on validation or test rows, alter labels, or change splits. Negatives include constructed decoys and some IMMREP-relabelled rows. Use `binding_flag` as given.

## Model

Keep `Model()`, `Dataset(path)`, `main()`, and `forward(peptide, hla, tcr)`. `Dataset.data` stays in original CSV order with the original inputs and labels. `forward` returns one finite binding probability in [0, 1] per row. Select eligible full-chain rows for training. Do not crop sequences.

The training loss may change. The comparable validation loss is the probability MSE from the fixed evaluator. Choose the next parent using that loss together with overall AUROC and AUPRC, peptide-macro AUROC and AUPRC, and the seen/unseen groups. A lower MSE alone is not a reason to adopt a parent. Do not claim an improvement the evaluation did not measure.

Do not evaluate the test split during the search.
