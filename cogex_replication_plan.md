# COGEX Replication Plan: Finetuning for Code Generation and Emulated Execution

## 1. Paper Overview

### 1.1 Core Idea
COGEX (Code Generation and Emulated EXecution) is a paradigm that trains language models to:
1. **Generate pseudo-programs** given a natural language instruction and optional input
2. **Emulate execution** of those programs, including undefined "leaf" functions
3. **Output structured results** containing intermediate reasoning steps and final answers

The key insight is that LMs can execute code *without a Python interpreter* by leveraging their latent knowledge to fill in undefined function implementations. This extends code-based reasoning to "soft" reasoning tasks (commonsense, social reasoning) that don't have clean algorithmic solutions.

### 1.2 Why This Matters
Traditional program synthesis requires fully-defined, compilable code. COGEX allows:
- Pseudo-programs with underspecified functions (e.g., `identify_ending(word)` without implementation)
- The LM's knowledge fills semantic gaps during emulated execution
- Applicable to tasks where pure algorithmic solutions are impossible

### 1.3 What We're Replicating
**Scope**: The supervised finetuning (SFT) step only—training a base LM on the COGEX dataset to produce a model capable of generating and emulating pseudo-programs.

**Not in scope**: 
- Dataset generation (already available on HuggingFace)
- COTACS program search (downstream task adaptation)
- Full benchmark evaluation

---

## 2. Dataset Specification

### 2.1 Source
The COGEX dataset is derived from the Alpaca instruction-tuning dataset via GPT-4 conversion (August 2023). It has been downloaded to ./data/data.json .

### 2.2 Dataset Size
- **Training**: ~52,000 examples (full Alpaca conversion)
- **Validation**: 2,000 examples randomly sampled from training set

### 2.3 Data Format
Each instance has this structure:

```
### Instruction
{task_description}
### Input
{optional_input_argument}
### Code
def {function_name}({args}):
    """
    {docstring}
    """
    # Step 1: {NL description}
    {variable} = {undefined_function_call}
    # Step 2: {NL description}
    {variable2} = {another_call}
    ...
    return {
        '{intermediate_key}': {intermediate_value},
        'answer': {final_answer}
    }

>>> {function_name}({concrete_args})
### Output
{
    '{intermediate_key}': {computed_intermediate},
    'answer': {computed_answer}
}
```

### 2.4 Key Data Properties
1. **Programs are general-purpose**: `pluralize_word(word)` not `pluralize_corpus()`
2. **Functions are intentionally undefined**: Only name + docstring, no implementation
3. **Output includes intermediates**: Forces model to follow reasoning plan in comments
4. **Pythonic data structures**: Inputs/outputs converted to strings, lists, dicts, integers

## 3. Model Configuration

### 3.1 Base Models
| Model | Parameters | HuggingFace ID |
|-------|-----------|----------------|
| Llama-2 7B | 7B | `Llama-2-7b-hf` |
| Llama-2 13B | 13B | `Llama-2-13b-hf` |
| Llama-3.1 8B | 8B | `Llama-3.1-8B` |

The original paper only uses the Llama-2 models, however, I want to test Llama 3.1.

### 3.2 LoRA Configuration
The paper uses parameter-efficient finetuning via LoRA:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `r` (rank) | 16 | Low-rank dimension |
| `lora_alpha` | 32 | Typically 2×r, paper doesn't specify—use default |
| `lora_dropout` | 0.05 | Specified in paper |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj"]` | Query, key, value, output matrices |
| `bias` | "none" | Standard practice |
| `task_type` | "CAUSAL_LM" | Autoregressive generation |

### 3.3 Training Hyperparameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 5 | Fixed |
| Batch size | 32 | Global batch size |
| Learning rate | 3e-4 | 0.0003 |
| LR scheduler | Cosine | Standard practice (paper doesn't specify) |
| Warmup | 3-5% of steps | Standard practice (paper doesn't specify) |
| Optimizer | AdamW | Standard (paper doesn't specify) |
| Weight decay | 0.0 | Typical for LoRA |
| Max sequence length | 2048 | Llama-2 context window (may need adjustment based on data) |
| Gradient checkpointing | True | For memory efficiency |
| Mixed precision | bf16 | For GPU specs |

### 3.4 Checkpoint Selection
- Save checkpoints each epoch (or more frequently)
- **Selection criterion**: Lowest perplexity on validation set
- This is cross-entropy loss, not task accuracy

---

## 4. Handling Input/Output Masking (Important!)
For instruction-tuning, you typically only compute loss on the **output** portion, not the instruction. However, for COGEX, the "output" includes:
- The generated code (model should learn to generate this)
- The function call
- The execution output

**Decision point**: Mask loss on instruction+input only, or compute loss on full sequence?

Paper doesn't specify. Conservative approach: mask instruction/input, compute loss on code+output.

```python
def create_labels_with_masking(input_ids, tokenizer, instruction_end_marker="### Code"):
    """Create labels that mask the instruction portion."""
    labels = input_ids.clone()
    
    # Find where instruction ends (before ### Code)
    # Set labels to -100 (ignored in loss) for instruction portion
    
    # Implementation depends on tokenizer and exact format
    # This is a simplification—actual implementation needs care
    
    return labels
```

## 5. Hardware and Performance Expectations

### 5.1 Reference Hardware (Paper)
- 128GB RAM
- 2× NVIDIA A6000 (48GB each)

### 5.2 Training Time (Paper)
| Model | Time |
|-------|------|
| 7B | ~12 hours |
| 13B | ~20 hours |

### 5.3 Memory Estimation
With LoRA + gradient checkpointing + bf16:
- 7B: ~24-30GB VRAM per GPU (fits on single A6000)
- 13B: ~40-48GB VRAM per GPU (may need model parallelism)

## 6. Verification and Sanity Checks

### 6.1 Pre-Training Checks
1. **Dataset verification**: 
   - Print 5 random examples, verify format
   - Check sequence length distribution
   - Verify no data leakage between train/val

2. **Model verification**:
   - `model.print_trainable_parameters()` should show ~0.1-0.5% trainable
   - Verify LoRA applied to correct modules

3. **Tokenization verification**:
   - Decode tokenized examples, verify reconstruction
   - Check for truncation issues

### 6.2 During Training Checks
1. **Loss curve**: Should decrease steadily
2. **Validation loss**: Should track training loss (if diverges, overfitting)
3. **Learning rate**: Verify warmup + decay schedule

### 6.3 Post-Training Checks
Generate outputs on held-out examples:
```python
prompt = """### Instruction
Pluralize the given word
### Input
Corpus
### Code"""

outputs = model.generate(
    tokenizer(prompt, return_tensors="pt").input_ids,
    max_new_tokens=512,
    temperature=0.7,
)
print(tokenizer.decode(outputs[0]))
```

Verify:
- Generates syntactically valid Python
- Function includes undefined helper calls
- Output dictionary has 'answer' key
- Intermediate steps are populated

## 7. Quick Reference: Key Paper Values

| What | Value |
|------|-------|
| Base model | Llama-2 7B/13B |
| LoRA rank | 16 |
| LoRA dropout | 0.05 |
| LoRA targets | q, k, v, o projections |
| Epochs | 5 |
| Batch size | 32 |
| Learning rate | 3e-4 |
| Validation size | 2,000 |
| Selection metric | Lowest validation perplexity |
| Training time (7B) | ~12 hours |
| Training time (13B) | ~20 hours |
| Hardware | 2× A6000 48GB |
