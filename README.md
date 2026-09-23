# Character-Level Language Models — From Scratch

Building character-level language models step by step, starting from simple counting
and moving toward neural networks. Each model is trained on a dataset of ~32,000 names
and learns to generate new, name-like words one character at a time.

The core task at every stage is the same one every large language model solves:
**given the context so far, predict a probability distribution over the next token.**

---

## Contents

| Part | Folder | Model | Result (avg. NLL) | Status |
|---|---|---|---|---|
| 1 | [`part1-bigram/`](./part1-bigram) | Bigram — counting and a single-layer neural net | 2.454 (counting) · 2.490 (neural net) | ✅ Complete |
| 2 | `part2-mlp/` | Multilayer perceptron with learned character embeddings | — | 🔄 In progress |

Baseline for comparison: uniform guessing over 27 characters = ln(27) ≈ 3.296. Lower is better.

Each part's folder has its own README with the full implementation details, results, and what I learned.

---

## Progression

- **Part 1 — Bigram:** predicts the next character from only the previous one. Built twice —
  by counting and as a one-layer neural net — to show that gradient descent learns the same
  table that counting computes directly.
- **Part 2 — MLP:** extends the context to several previous characters and learns a
  vector embedding for each character, which counting can't scale to.

---

## How to Run

```bash
git clone https://github.com/its-Ravi-Singh/char-level-language-model.git
cd char-level-language-model/part1-bigram
pip install torch matplotlib
jupyter notebook bigram.ipynb
```

---

## Tech Stack

`Python` · `PyTorch` · `Matplotlib`

---

*Contact: raviraja@buffalo.edu · [LinkedIn](https://linkedin.com/in/ravi-rajaram-singh-47551a206)*
