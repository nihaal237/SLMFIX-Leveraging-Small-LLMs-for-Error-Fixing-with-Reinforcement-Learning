# SLMFix 🔧
### Leveraging Small Language Models for Error Fixing with Reinforcement Learning

> A small language model (0.5B) trained with PPO-based RL to automatically fix broken SQL and Bash code — rewarded on static validity and AST/semantic similarity.

---

## Overview

LLMs often generate plausible-looking but broken code. **SLMFix** addresses this by training a lightweight *fixer* model to correct the mistakes of a larger *proposer* model — using reinforcement learning rather than supervised imitation.

The pipeline works in three stages:

```
Natural Language Query
        │
        ▼
┌─────────────────────┐
│  Proposer (7B)      │  Qwen-2.5-Coder-7B-Instruct
│  Generates SQL/Bash │  (4-bit quantised)
└─────────────────────┘
        │  broken code + error
        ▼
┌─────────────────────┐
│  Fixer (0.5B)       │  Qwen-2.5-Coder-0.5B
│  Repairs the code   │  trained with PPO
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Reward Signal      │  Static validation +
│                     │  AST / Semantic similarity
└─────────────────────┘
```

This project reproduces and extends the [SLMFix paper (Fu et al., 2025)](https://arxiv.org/abs/2511.19422), scoped to **SQL** and **Bash** code generation.

---

## Datasets

| Language | Dataset | Train | Dev | Test |
|----------|---------|-------|-----|------|
| SQL | [Spider 1.0](https://yale-lily.github.io/spider) | 7,000 | 1,034 | 2,147 |
| Bash | [NL2Bash](https://huggingface.co/datasets/jiacheng-ye/nl2bash) | 8,090 | 606 | 606 |

---

## Reward Function

The PPO reward is a **weighted combination** of two signals:

### SQL
- **Static validity** — SQLGlot parses the query without error → `1.0`
- **Semantic similarity** — Reciprocal of AST edit distance between predicted and gold SQL (via SQLGlot)

### Bash
- **Static validity** — bashlex parses the command without error
- **Semantic similarity** — Option-value dictionary matching: atomic commands are parsed into `{flag: argument}` dicts and compared against ground truth

---

## Failure Taxonomy

Generated code is classified before being stored as a training triple:

| Type | Label | Description |
|------|-------|-------------|
| A | `static_failure` | SQLGlot / bashlex rejects the query syntactically |
| B | `runtime_failure` | Passes static check but fails at SQLite execution |
| C | `semantic_mismatch` | Executes correctly but AST similarity < 0.80 threshold |

---

## Setup & Usage

### Requirements
```bash
pip install torch transformers trl peft sqlglot bashlex
```

### Running in Google Colab
1. Mount your Google Drive
2. Place datasets under `SLMFix_Project/` as structured above
3. Open `SLMFixProject.ipynb` and run cells sequentially

## Models

| Role | Model | Size | Quantisation |
|------|-------|------|-------------|
| Proposer | Qwen-2.5-Coder-7B-Instruct | 7B | 4-bit (BnB) |
| Fixer | Qwen-2.5-Coder-0.5B-Instruct | 0.5B | Full precision |
