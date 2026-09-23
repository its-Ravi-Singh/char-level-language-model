# Character-Level Language Models — From Scratch

Building character-level language models step by step, starting from simple counting
and moving toward neural networks. Each model is trained on a dataset of ~32,000
names and learns to generate new, name-like words one character at a time.

The core task at every stage is the same one every large language model solves:
**given the context so far, predict a probability distribution over the next token.**

---

## Contents

| Part | Folder | Model | Status |
|---|---|---|---|
| 1 | [`part1-bigram/`](./part1-bigram) | Bigram model — counting and a single-layer neural net | ✅ Complete |
| 2 | `part2-mlp/` | Multilayer perceptron with learned character embeddings | 🔄 In progress |

---

## Part 1 — Bigram Model

A bigram model predicts the next character using only the one character before it.

**What I implemented**
- A 27-token vocabulary: `a–z` plus a special `.` token marking the start and end of a name
- **Counting approach:** a 27×27 table where each cell stores how often character *j*
  follows character *i*, normalized row-wise into probabilities
- **Smoothing:** adding 1 to every count so no transition gets zero probability
  (which would make the loss infinite on any unseen pair)
- **Evaluation:** average negative log-likelihood over the dataset — lower is better
- **Sampling:** generating new names by repeatedly drawing the next character from
  the model's probability distribution until the end token appears
- **Neural-network approach:** the same model rebuilt as a single linear layer —
  one-hot encoded input → 27×27 weight matrix → softmax — trained with gradient
  descent to minimize the same negative log-likelihood

**Result**

Both approaches converge to the same average negative log-likelihood (≈ `<your loss>`),
which is the point: the neural net *learns* the same table that counting computes
directly. Weight regularization in the neural version plays the same role that
add-one smoothing plays in the counting version.

**Why it matters**

The neural formulation is the one that scales. Counting breaks down as soon as the
context grows beyond one character — the table size explodes — but the
gradient-based setup extends naturally to longer contexts, embeddings, and
eventually Transformers. Part 2 takes that next step.

---

## How to Run

```bash
git clone https://github.com/its-Ravi-Singh/char-level-language-model.git
cd char-level-language-model/part1-bigram
pip install torch matplotlib
jupyter notebook
```

---

## Tech Stack

`Python` · `PyTorch` · `Matplotlib`

---

*Contact: raviraja@buffalo.edu · [LinkedIn](https://linkedin.com/in/ravi-rajaram-singh-47551a206)*
