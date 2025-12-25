# D-RAG Phase 2 (CWQ) – Debugging + Implementation Changes Summary (2025-12-25)

This note summarizes the **material code and workflow changes** made while debugging and improving the Phase 2 (joint retriever+generator) training run, with a focus on **NemotronGenerator decoding/EOS behavior** and **retriever metrics stability**.

---

## 1) Root issues that were investigated/fixed

- **Generator “clipping/slicing” artifacts**: generation outputs were being sliced incorrectly (wrong prompt-length assumptions) leading to truncated / empty decoded answers.
- **Degenerate repetition / empty predictions**:
  - Some runs generated repeated first-token patterns (e.g., `D D D...`) instead of completing an entity.
  - Some runs generated **PAD** tokens and then produced empty strings after decoding/stripping.
- **EOS supervision correctness**: ensured the training labels actually include **`answer + EOS`** and that loss is computed on EOS.
- **Hybrid prompting generation bug**: generation with `inputs_embeds` on this Nemotron/Mamba hybrid caused broken autoregressive continuation (context not preserved correctly), requiring a custom generation loop.
- **Retriever loss balancing instability**: paper-aligned gradient-norm balancing could produce highly unstable weights in practice; we kept a stable static weight as default.

---

## 2) Files changed (high signal)

### `src/model/generator.py` – NemotronGenerator

- **Special token alignment / sanity checks**
  - Added a startup print of `tokenizer.eos_token_id`, `model.config.eos_token_id`, `tokenizer.pad_token_id`, etc.
  - Defensive syncing of tokenizer/config for EOS/PAD to avoid immediate-stop / never-stop / empty-decode failure modes.
- **Prompt boundary / tokenization alignment**
  - Adjusted the answer header formatting and ensured the supervised answer begins with a leading space when needed, so tokenization at the `"Answer:"` boundary is stable.
- **EOS supervision + diagnostics**
  - Fixed label construction so the supervised region is **`answer_ids + [eos_id]`**.
  - Added `DRAG_DEBUG_EOS=1` diagnostics to print supervised ids/decoded text and confirm EOS is the last supervised label.
  - Added optional `--eos_loss_weight` support to upweight EOS token loss during training (keeps objective as cross-entropy but reweighted).
- **Custom greedy generation for hybrid inputs**
  - Implemented a custom autoregressive loop for `generate()` that appends newly generated token embeddings to the context, avoiding the broken `inputs_embeds` autoregressive behavior observed with the base generation utilities.
  - Ensured EOS termination is honored and guarded against PAD-token generation.
- **Fixes from this session**
  - Repaired two indentation corruption points that prevented `generator.py` from importing/running (`IndentationError` at import-time and a broken `with torch.no_grad()` block inside the generation loop).

### `src/trainer/train_phase2.py` – joint training loop

- **Validation speed knob**: `--val_generation_limit N` to run free-generation metrics only on the first `N` validation examples, while still computing loss + retrieval metrics on all val examples.
- **Paper-style “hybrid selection” for validation/inference**
  - `top-k` cap (`--max_facts_cap`, paper uses 100) + probability threshold filtering (`--prob_threshold`, paper uses 0.01) with a top-1 fallback.
- **Loss weighting options**
  - Retained stable default **static** retriever auxiliary weight (`--ret_loss_weight`, default `0.1`).
  - Added optional **paper-aligned gradient-norm balancing** (`--use_grad_norm_balance`, `--rho`) and periodic diagnostics logging (`--log_grad_norms`, `--grad_norm_log_interval`).
- **Paper-style multi-answer formatting**
  - If `answer` is a list, it is joined as `" | "` for supervision (metrics still evaluate against the list).

### `src/utils/metrics.py`

- **More robust token-F1**
  - Improved tokenization fallback to handle concatenated/no-space outputs (e.g. `DavidDavid...`) so token-F1 isn’t artificially forced to 0.

### `scripts/debug_eos_supervision.py`

- Added/updated a diagnostic script to:
  - Print special token ids and tokenization details.
  - Validate `forward_paper_prompt` label construction with `DRAG_DEBUG_EOS=1`.
  - Load a trained LoRA adapter from `checkpoints_cwq_phase2_paperprompt/generator_best` for realistic checks.
  - Exercise the updated custom generation path to confirm EOS can be produced and that repetition isn’t caused by decoding-only issues.

### `README.md`

- Updated Phase 2 training examples to match the current CLI (removed stale `--k_facts` usage).
- Added a copy/paste **`nohup + disown`** command to run training in the background and keep it alive after closing the terminal.
- Documented CWQ Phase 2 train/val split files (`data/train_heuristics_cwq_train.jsonl`, `data/train_heuristics_cwq_val.jsonl`).

---

## 3) Operational notes / recommended defaults

- **Decoding**: Greedy decoding is kept as default (deterministic metrics; repetition should be addressed by EOS learning rather than sampling).
- **Retriever weighting**:
  - Default: `--ret_loss_weight 0.1` (stable).
  - Optional: `--use_grad_norm_balance --rho 0.9` (paper-aligned, but can be unstable depending on gradient-norm ratio swings).
- **Validation speed**:
  - Use `--val_generation_limit` to keep epoch validation fast while still tracking loss and retrieval metrics across the whole val set.

---

## 4) “Quick start” background training command (CWQ)

The canonical background command is now documented in `README.md` under **“Run Phase 2 in the background”**.

---

## 5) Dataset sanity checks (evaluation caveats)

During debugging we also verified:
- Train/val are loaded from separate paths (`--heuristics_path` vs `--val_heuristics_path`).
- There is a tiny overlap in question strings between CWQ train/val (small enough not to explain strong metrics by itself).
- A subset of validation questions contain the answer as a substring (dataset property), which can inflate EM/Hits@1.


