# Automated research projects

Reproducible release packages for automated AI-safety research projects. Each project folder is
self-contained: minimal cleaned code, a master Jupyter notebook, and a figure-generation entry point
that regenerate the main results on CPU from artifacts hosted on Hugging Face.

## Projects

- **[`cot-controllability-steering-vectors/`](cot-controllability-steering-vectors/)** — *A
  2,880-number steering vector gives a reasoning model the chain-of-thought control that fine-tuning
  does* (`gpt-oss-20b`). A single frozen-weights steering vector reproduces what a LoRA fine-tune does
  to the model's CoT controllability, including instruction-conditional formatting; mechanistically it
  makes late attention heads read the in-context instruction's format-specifier tokens. The package
  regenerates the five publication figures from released artifacts on CPU (no GPU, no model
  generation).
