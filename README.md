<div align="center">

# Marker–Aligned Subspace Separation for Implicit Sentiment Analysis (MASS)

[![Venue](https://img.shields.io/badge/Venue-SOICT%202026%20(Accepted)-blue?style=flat-square)](./paper.pdf)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat-square&logo=adobeacrobatreader&logoColor=white)](./paper.pdf)
[![Colab Notebook](https://img.shields.io/badge/Notebook-Google%20Colab-f9ab00?style=flat-square&logo=googlecolab&logoColor=white)](./MASS_run_notebook.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](./requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](./requirements.txt)
[![Backbone](https://img.shields.io/badge/Backbone-Flan--T5%20(Base%20%26%20XXL)-ffcc00?style=flat-square)](https://huggingface.co/google/flan-t5-base)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](./LICENSE)

**Official PyTorch implementation of the SOICT 2026 paper:**  
📄 **[Marker–Aligned Subspace Separation for Implicit Sentiment Analysis](./paper.pdf)**

[Overview](#overview) •
[Repository Structure](#repository-structure) •
[Installation](#installation) •
[Datasets](#datasets) •
[Main Results](#main-results) •
[Usage](#usage) •
[Outputs](#outputs) •
[License](#license)

</div>

---

## Overview

<div align="center">
  <img src="./figure.png" alt="Overall architecture of the proposed MASS framework" width="100%">
  <br>
  <em><b>Figure 1:</b> The overall architecture of our proposed MASS framework.</em>
</div>

<br>

**MASS** (*Marker-Aligned Subspace Separation*) is a supervised fine-tuning framework for **Aspect-Based Sentiment Analysis (ABSA)** and **Implicit Sentiment Analysis (ISA)** built on the Flan-T5 encoder. Unlike approaches that rely on costly LLM-driven synthetic data augmentation or error-prone multi-hop generation chains, MASS directly addresses two fundamental bottlenecks in implicit sentiment reasoning—*limited focus on the target aspect term* and *frequent confusion at the Neutral–Polar boundary*:

* **Aspect Marker & Structured Two-Step Prompting**: Wraps the target aspect term with sentinel tokens (`<extra_id_0>` and `<extra_id_1>`) inside a structured two-step instruction prompt and extracts the aspect-focused contextual representation at the `<extra_id_0>` marker position solely from the Flan-T5 encoder.
* **Subjectivity–Polarity Subspace Decomposition**: Projects this marker representation into two separate representation subspaces—*Subjectivity* (`Neutral` vs. `Polar`) and *Polarity* (`Positive` vs. `Negative`)—optimized jointly via class-balanced cross-entropy losses and a Frobenius-norm orthogonality regularization penalty, followed by hierarchical inference.

---

## Repository Structure

```text
MASS/
├── paper.pdf                # Paper PDF
├── figure.png               # Figure 1 architecture diagram (high-resolution PNG)
├── config.yaml              # Full experiment, model, hyperparameter, and ablation configurations
├── main.py                  # CLI entrypoint for training, evaluation, and multi-seed aggregation
├── MASS_run_notebook.ipynb  # Self-contained Google Colab notebook (Flan-T5-Base, Flan-T5-XXL & Ablations)
├── requirements.txt         # Exact Python package versions from Google Colab
├── data/                    # SemEval-2014 Task 4 ABSA datasets with implicit sentiment labels
│   ├── laptops/
│   │   ├── Laptops_Train_v2_Implicit_Labeled.xml
│   │   └── Laptops_Test_Gold_Implicit_Labeled.xml
│   └── restaurants/
│       ├── Restaurants_Train_v2_Implicit_Labeled.xml
│       └── Restaurants_Test_Gold_Implicit_Labeled.xml
└── src/
    ├── config.py            # Configuration schema and YAML resolver
    ├── data.py              # XML parser, prompt builder, and target-centered tokenization dataset
    ├── evaluation.py        # Hierarchical decoding, metrics (Acc. / MF1), and seed aggregation
    ├── labels.py            # Sentiment polarity label mappings
    ├── logging_utils.py     # Timestamped experiment logger
    ├── models.py            # MassModel (Flan-T5 encoder, subspace projections, orthogonality loss)
    └── training.py          # End-to-end training loop and checkpoint management
```

---

## Installation

Install the required dependencies (tested on Google Colab with CUDA 12.8):

```bash
pip install -r requirements.txt
```

---

## Datasets

We evaluate on the **Laptop** and **Restaurant** benchmarks from SemEval-2014 Task 4, extended with boolean `implicit_sentiment` annotations by Li et al. (2021). Following prior work, conflict-labeled instances are removed from both training and test sets.

| Domain | Split | Total | Positive | Negative | Neutral | Implicit (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Laptop** | Train | 2,313 | 987 | 866 | 460 | 716 (30.96%) |
| | Test | 638 | 341 | 128 | 169 | 175 (27.43%) |
| | **Total** | **2,951** | **1,328** | **994** | **629** | **891 (30.19%)** |
| **Restaurant** | Train | 3,602 | 2,164 | 805 | 633 | 1,030 (28.60%) |
| | Test | 1,120 | 728 | 196 | 196 | 267 (23.84%) |
| | **Total** | **4,722** | **2,892** | **1,001** | **829** | **1,297 (27.47%)** |

---

## Main Results

Main experimental comparison across `Flan-T5-Base` (250M) and LLM (`8B–11B`) backbones, averaged over 3 random seeds. Within each backbone group, the best score is shown in **bold** and the second-best is <u>underlined</u>.

| Method | Rest. All-Acc. | Rest. All-MF1 | Rest. ISA-Acc. | Rest. ISA-MF1 | Lap. All-Acc. | Lap. All-MF1 | Lap. ISA-Acc. | Lap. ISA-MF1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Flan-T5-Base + THOR | 86.16 | 77.98 | 64.42 | 64.69 | 80.72 | 76.70 | 73.14 | 69.07 |
| Flan-T5-Base + Vanilla Inst-FT | 86.52 | 79.10 | 64.79 | 65.31 | 81.03 | 77.39 | 75.43 | 71.29 |
| Flan-T5-Base + RVISA<sub>v</sub> | 86.43 | 78.49 | 65.92 | – | 80.72 | 76.49 | 73.71 | – |
| Flan-T5-Base + MT-ISA<sub>O</sub> | 85.89 | 77.85 | 66.29 | – | 81.98 | 78.10 | 75.43 | – |
| Flan-T5-Base + RVISA<sub>g</sub> | 86.61 | 78.92 | 66.67 | – | 81.19 | 77.13 | 75.43 | – |
| Flan-T5-Base + MT-ISA<sub>I</sub> | <u>88.21</u> | <u>82.45</u> | <u>70.41</u> | – | **82.91** | <u>79.86</u> | <u>80.57</u> | – |
| **Flan-T5-Base + MASS (Ours)** | **88.35** | **83.47** | **74.87** | **73.21** | <u>82.19</u> | **80.34** | **81.67** | **75.08** |
| Llama-3.1-8B-IT + Vanilla Inst-FT | 89.11 | 82.53 | 76.78 | 77.07 | 82.60 | 78.74 | 77.14 | 75.17 |
| Qwen3-8B + Vanilla Inst-FT | <u>90.62</u> | <u>85.03</u> | <u>77.53</u> | <u>77.15</u> | <u>83.39</u> | <u>80.12</u> | <u>80.57</u> | <u>77.60</u> |
| **Flan-T5-XXL + MASS (Ours)** | **91.16** | **86.27** | **82.77** | **81.88** | **85.89** | **82.95** | **87.43** | **85.16** |

> **Notation:**
> * **Domains**: `Rest.` = Restaurant, `Lap.` = Laptop.
> * **Evaluation Subsets**: `All` = Evaluation on the full test set (both explicit and implicit sentiment instances), `ISA` = Evaluation on the implicit-only test set subset.
> * **Metrics**: `Acc.` = Accuracy (%), `MF1` = Macro-F1 (%).

---

## Usage

### 1. Run Main Experiments

Train and evaluate **MASS** with `Flan-T5-Base` across all 3 seeds on both datasets:

```bash
python main.py --variant base --dataset restaurants laptops
```

Train and evaluate **MASS** with `Flan-T5-XXL` (`bfloat16` + LoRA):

```bash
python main.py --variant xxl --dataset restaurants laptops
```

Run a single dataset and seed:

```bash
python main.py --variant base --dataset restaurants --seed 42
```

### 2. Run Ablation & Marker-Representation Variants

Run all 6 ablation and marker-representation variants via the `--ablation` CLI argument:

```bash
python main.py \
  --variant base \
  --dataset restaurants laptops \
  --ablation wo_aspect_marker wo_orthogonality wo_two_subspaces wo_class_weights end_marker mean_span_pooling
```

### 3. Preview Resolved Configuration

Inspect the full resolved configuration without starting training:

```bash
python main.py --variant base --dataset restaurants --ablation full --seed 42 --print-config
```

### 4. Interactive Google Colab Notebook

Open [`MASS_run_notebook.ipynb`](./MASS_run_notebook.ipynb) on Google Colab to run **Flan-T5-Base**, **Flan-T5-XXL**, or any **Ablation** variant interactively in a single self-contained notebook.

---

## Outputs

Each run saves its artifacts under `outputs/{dataset}/{variant}/seed_{seed}/` (or `outputs/{dataset}/{variant}/{ablation}/seed_{seed}/` for non-default ablations):

* `checkpoints/best_model.pt` — Model weights from the epoch achieving the highest ISA Macro-F1.
* `results/best_metrics.json` — Best evaluation metrics (`all_accuracy`, `all_macro_f1`, `isa_accuracy`, `isa_macro_f1`) and resolved configuration.
* `results/epoch_metrics.jsonl` — Per-epoch training losses and test metrics.
* `predictions/full_test_predictions.csv` — Instance-level predictions on the full test set (`All`).
* `predictions/isa_test_predictions.csv` — Instance-level predictions on the implicit-only subset (`ISA`).
* `aggregate_metrics.json` — Mean and standard deviation aggregated across all specified seeds.

---

## License

This project is released under the [MIT License](./LICENSE).
