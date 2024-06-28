# SemEval-2024 Task 6: SHROOM - a Shared-task on Hallucinations and Related Observable Overgeneration Mistakes

## Overview of the task

The modern NLG landscape is plagued by two interlinked problems: On the one hand, our current neural models have a propensity to produce inaccurate but fluent outputs; on the other hand, our metrics are most apt at describing fluency, rather than correctness. This leads neural networks to “hallucinate”, i.e., produce fluent but incorrect outputs that we currently struggle to detect automatically. For many NLG applications, the correctness of an output is however mission-critical. For instance, producing a plausible-sounding translation that is inconsistent with the source text puts in jeopardy the usefulness of a machine translation pipeline. With our shared task, we hope to foster the growing interest in this topic in the community.

Participants will be asked to perform binary classification to identify cases of fluent overgeneration hallucinations in two different setups: model-aware and model-agnostic tracks.
Simply put, participants must detect grammatically sound outputs which contain incorrect or unsupported semantic information, inconsistent with the source input, with or without having access to the model that produced the output.

To that end, we will provide participants with a collection of checkpoints, inputs, references and outputs of systems covering three different NLG tasks: definition modeling (DM), machine translation (MT) and paraphrase generation (PG), trained with varying degrees of accuracy. The development and test set will provide binary annotations from at least five different annotators and a majority vote gold label.

## Evaluation protocol

Submissions will be divided into two tracks: a model-aware track, where we provide a checkpoint to a model publically available on HuggingFace for every datapoint considered, and a model-agnostic track where we do not. We highly encourage participants to make use of model checkpoints in creative ways.
For both tracks, all participants' submissions will be evaluated using two criteria:

1. the accuracy that the system reached on the binary classification; and
2. the Spearman correlation of the systems' output probabilities with the proportion of the annotators marking the item as overgenerating

## Repository structure

```bash
.
├── LICENSE
├── README.md
├── baseline.ipynb          # Baseline from organizers
├── check_output.py         # Script to check output format for submission
├── data.ipynb              # Loading and displaying data
├── ensemble.ipynb          # Training of ensemble with Catboost metamodel
├── models.ipynb            # Training of transormer-based models
├── output
│   ├── catboost            # Output from ensemble
│   ├── deberta_inverse_nli # Output from DeBERTa (inverse NLI)
│   ├── deberta_nli         # Output from DeBERTa (NLI)
│   ├── openchat_chain_poll # Output from OpenChat
│   ├── roberta_inverse_nli # Output from RoBERTa (inverse NLI)
│   ├── roberta_nli         # Output from RoBERTa (NLI)
│   ├── st_roberta_nli      # Output from RoBERTa from sentence-transformers lib.
│   ├── t5_inverse_nli      # Output from T5 (inverse NLI)
│   └── t5_nli              # Output from T5 (NLI)
├── requirements.txt
└── score.py                # Script to score output
```

### About output checker script

The output checker script is intended as a general first pass to ensure that the model's output generally correspond to the requirements of the scoring program. Note that it may not flag all potential issues.

To evaluate the well-formedness of a validation-split output, use the `--is_val` flag.
Provide paths to directories containing their submission file(s), rather than path to files.

### About scoring script

The scorer script requires the reference data to compute scores.
Note that the scorer script may reject a submission that was marked as correct by the output format checker script.

To evaluate the performances on validation data, use the `--is_val` flag. The reference data file names must be modified to not include version numbering when present.
Provide paths to directories containing their submission file(s) and the corresponding reference files.

## Referencies

- [Our arcitcle about the developed system](https://aclanthology.org/2024.semeval-1.42)
- [Organizers' article about the task](https://aclanthology.org/2024.semeval-1.273)
