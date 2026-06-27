"""CoT-controllability steering vectors -- minimal release package.

Public surface:
  * ``cot_steering.figures`` -- regenerate fig1-fig5 from released summary artifacts.
  * ``cot_steering.artifacts`` -- locate artifacts (Hugging Face, local fallback).
  * ``cot_steering.instructions`` -- the 25-instruction CoT-control suite + scorers + splits.
  * ``cot_steering.steering`` -- load a steering ``.npz`` and the residual-stream apply hook.
"""

__all__ = ["figures", "artifacts", "instructions", "steering"]
__version__ = "1.0.0"
