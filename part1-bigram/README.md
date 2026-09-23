# Part 1 — Bigram Character-Level Language Model

A language model that predicts the next character from only the previous one, built two
ways — by direct counting, and as a single-layer neural network trained with gradient
descent — to show that both arrive at the same model.

Dataset: `names.txt` (~32K names) → **228,146 character bigrams** for training.

---

## 1. Counting approach

- **Vocabulary:** 27 tokens — `a–z` plus `.`, a special token that marks both the start
  and the end of a name
- **Count table:** a 27×27 tensor `N`, where `N[i, j]` = how many times character `j`
  follows character `i` in the dataset (visualized as a heatmap in the notebook)
- **Smoothing:** `P = (N + 1)`, so no bigram has zero probability — otherwise any unseen
  pair gives `log(0) = -inf` loss. Adding a larger constant gives a smoother, more uniform
  distribution; a smaller one keeps it peaked.
- **Normalization:** each row divided by its sum → `P[i]` is the probability distribution
  over the next character given character `i`
- **Sampling:** start at `.`, repeatedly draw the next character with `torch.multinomial`
  until `.` comes up again

**Evaluation — average negative log-likelihood**

Maximizing the likelihood of the data = maximizing the log-likelihood (log is monotonic)
= minimizing the negative log-likelihood. Averaging over all bigrams gives the loss.

---

## 2. Neural-network approach

The same model, learned instead of counted:

```
one-hot(prev char)  →  × W (27×27)  →  logits  →  exp  →  normalize  →  P(next char)
   [N, 27]                              "log-counts"     "counts"       (softmax)
```

- **Parameters:** a single 27×27 weight matrix `W`, randomly initialized
- **Loss:** mean negative log-likelihood of the correct next character, plus
  `0.01 · mean(W²)` regularization
- **Training:** 100 steps of full-batch gradient descent, learning rate 50

**Why it matches counting:** multiplying a one-hot vector by `W` just selects one row of
`W`. After `exp` and normalization, that row *is* the probability distribution for the
next character — the same row that counting builds directly. Regularization pulls `W`
toward zero, which pushes the distribution toward uniform — the same effect add-one
smoothing has on the count table.

---

## Results

| Model | Avg. negative log-likelihood (full dataset, 228,146 bigrams) |
|---|---|
| Uniform guessing (baseline) | ln(27) ≈ 3.296 |
| Counting, add-1 smoothing | **2.454** |
| Neural net, 100 steps | **2.490** *(includes regularization term)* |

The neural net lands just above the counting model: its loss includes the regularization
penalty, and 100 gradient steps haven't fully converged. More steps close the gap — both
are approximating the same optimal bigram table.

**Same seed, near-identical samples:**

| Counting model | Neural net |
|---|---|
| cexze. | cexze. |
| momasurailezitynn. | momasurailezityha. |
| konimittain. | konimittain. |
| llayn. | llayn. |
| ka. | ka. |

---

## What I learned

- A language model is just a probability distribution over the next token, conditioned
  on context — here the context is one character.
- Negative log-likelihood is the natural loss for this: it heavily punishes assigning
  low probability to what actually came next.
- Counting stops working once context gets longer (27² → 27³ → … table sizes), but the
  neural formulation doesn't — which is why Part 2 moves to a neural network with a
  longer context and learned embeddings.

---

## Run it

```bash
cd part1-bigram
pip install torch matplotlib
jupyter notebook bigram.ipynb
```
