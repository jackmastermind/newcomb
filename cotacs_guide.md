# CoTACS: COGEX Task Adaptation via Code Search

## Overview

CoTACS is a program search procedure that adapts a pre-trained CoGEX model to a new task **without any gradient updates or parameter changes**. Instead of fine-tuning, you search for one or more pseudo-programs that, when reapplied across all instances of a dataset, maximize task performance.

The core insight: since CoGEX models accept argument-accepting pseudo-programs, you can decouple *program generation* from *program execution*. This lets you find a single program $P_{\text{task}}$ that generalizes across a problem class, then apply the emulator:

$$f_{\text{emulator}}(I, A_i, P_{\text{task}}) \rightarrow C_i \rightarrow O_i$$

for each instance $A_i$.

---

## Prerequisites

Before running CoTACS, you need:
- A fine-tuned CoGEX model (7B or 13B parameters, trained on the CoGEX dataset)
- A target dataset $D = \{(a_1, o_1), (a_2, o_2), \ldots\}$ with input arguments and ground-truth outputs
- A task instruction $I_{\text{task}}$ describing what the model should do
- A task metric $\delta$ (e.g., exact match accuracy)

---

## Algorithm

### Step 1: Split the Dataset

Partition your dataset into:
- **Training set** ($n$ examples, typically $n = 300$): Used to generate candidate programs and filter low-quality ones
- **Development set** (remaining examples): Used to rank programs and select the best

```python
train_set = random_sample(D, n=300)
dev_set = D - train_set  # e.g., 700 examples if |D| = 1000
```

### Step 2: Generate Candidate Programs

For each training instance $(a_i, o_i)$, prompt the CoGEX model to generate a program:

```python
programs = []
for (a_i, o_i) in train_set:
    p_i = cogex_model.generate(I_task, a_i, temperature=0.7)
    programs.append(p_i)
```

Use a **sampling temperature of 0.7** during generation to encourage diversity in candidate programs.

### Step 3: Filter by Training Performance

Evaluate each candidate program on the *entire* training set. Discard programs that fall below a minimum performance threshold $\alpha$:

```python
filtered_programs = []
for p in programs:
    train_perf = evaluate(p, train_set, cogex_model, metric=delta)
    if train_perf >= alpha:
        filtered_programs.append(p)
    else:
        # Optionally resample until threshold met or max retries exceeded
        p_new = resample_until_threshold(cogex_model, I_task, a_i, alpha)
        filtered_programs.append(p_new)
```

The threshold $\alpha$ is user-defined and task-dependent. If no program achieves $\alpha$ after a fixed number of attempts, use the best-performing candidate sampled so far.

### Step 4: Rank on Development Set

Evaluate all filtered programs on the development set and rank by performance:

```python
ranked = sorted(
    filtered_programs,
    key=lambda p: evaluate(p, dev_set, cogex_model, metric=delta),
    reverse=True
)
```

### Step 5: Select Top-$k$ Programs

Retain the top $k$ programs. The paper experiments with $k \in \{1, 3\}$:

```python
P_D = ranked[:k]
```

For $k > 1$, use **majority voting** at test time: run the model with each program and take the most common answer.

---

## Evaluation Function

The `evaluate` function applies a program to all instances in a dataset and computes the average metric:

```python
def evaluate(program, dataset, model, metric):
    scores = []
    for (a_i, o_i) in dataset:
        # Run model in emulator mode with fixed program
        call, output = model.emulate(I_task, a_i, program, temperature=0.05)
        score = metric(output, o_i)
        scores.append(score)
    return mean(scores)
```

Use **temperature 0.05** during evaluation for near-deterministic outputs.

---

## Pseudocode Summary

```
Input:  CoGEX model f, Dataset D, Instruction I, 
        n (candidates), α (threshold), k (programs to keep), δ (metric)
Output: P_D (set of k optimal programs)

1.  Programs ← ∅
2.  TrainSet ← RandomSample(D, n)
3.  DevSet ← D \ TrainSet

4.  for (a_i, o_i) in TrainSet:
5.      p_i ← f(I, a_i)                    // Generate program
6.      while Evaluate(p_i, TrainSet) < α:
7.          p_i ← f(I, a_i)                // Resample if below threshold
8.      Programs.add(p_i)

9.  P_D ← argmax over subsets {p_1,...,p_k} ⊆ Programs
              of Σ Evaluate(p_j, DevSet)

10. return P_D
```

---

## Hyperparameter Recommendations

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| $n$ (training examples) | 300 | For generating candidates |
| Total dataset size | 1000 | 300 train + 700 dev |
| Generation temperature | 0.7 | Encourages program diversity |
| Evaluation temperature | 0.05 | Near-deterministic answers |
| $k$ (programs to keep) | 1 or 3 | $k=3$ with majority vote often helps |
| $\alpha$ (threshold) | Task-dependent | Start with ~0.5× baseline performance |

---

## Efficiency Considerations

**Search space reduction**: The paper shows you can reduce to ~50 training examples and ~10 program candidates while staying within 2 points of full performance on most tasks. This dramatically cuts compute costs.

**Comparison to fine-tuning**: CoTACS stores only a program string (a few KB) versus an entire model checkpoint. It outperforms fine-tuning when you have fewer than ~500 examples, making it ideal for low-to-medium shot scenarios.

---

## Test-Time Inference

Once you have $P_D$, apply it to new test instances:

```python
def predict(test_input, P_D, model, I_task):
    if len(P_D) == 1:
        _, output = model.emulate(I_task, test_input, P_D[0], temperature=0.05)
        return output['answer']
    else:
        # Majority vote for k > 1
        answers = []
        for p in P_D:
            _, output = model.emulate(I_task, test_input, p, temperature=0.05)
            answers.append(output['answer'])
        return majority_vote(answers)
```

---

## When to Use CoTACS vs. Alternatives

| Scenario | Recommendation |
|----------|----------------|
| Few labeled examples (<500) | CoTACS |
| Many labeled examples (>1000) | Fine-tuning |
| Tasks benefiting from programmatic structure (math, symbolic) | CoTACS with code programs |
| Pure NLP classification | CoTACS still works, but gains are smaller |
| Need to avoid storing model checkpoints | CoTACS |

---

## Common Pitfalls

1. **Program quality variance**: On some tasks (e.g., Word Sorting), generated programs have similar quality, so even a few candidates suffice. On others (e.g., CoinFlip), variance is high—sample more candidates.

2. **Emulation errors**: The LM may produce outputs inconsistent with the program logic. Inspect intermediate outputs in the returned dictionary to debug.

3. **Overfitting to dev set**: If your dev set is small, the top-ranked program may not generalize. Use larger dev sets when possible.

---

## Example: Adapting to Emotion Classification

```python
I_task = "Classify the emotion expressed in the given text."
D = load_emotion_dataset()  # list of (text, label) pairs

# Run CoTACS
P_D = cotacs_search(
    model=cogex_model,
    dataset=D,
    instruction=I_task,
    n=300,
    alpha=0.4,
    k=3,
    metric=exact_match
)

# Save just the program strings
save_programs(P_D, "emotion_programs.json")

# At test time
for test_text in test_data:
    prediction = predict(test_text, P_D, cogex_model, I_task)
```

The entire "adaptation" is stored in `emotion_programs.json`—no model weights saved.
