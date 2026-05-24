"""Dataset constitution module for warmup-phase supervised training.

Self-play alone fails to bootstrap a Morris agent (95%+ draws, value collapse).
This module produces a corpus of minimax-vs-minimax games with rich-heuristic
evaluation and ε-greedy + opening-random diversification, which the supervised
phase consumes as training targets before resuming self-play.
"""
