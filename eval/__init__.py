"""
Evaluation harness.

Every number this package produces must satisfy five conditions before it may
appear in a paper. They are listed in RETRACTION.md and enforced here:

1. Queries pass the contamination gate (eval/overlap.py).
2. Relevance is graded 0-3 and pooled across conditions (eval/metrics.py).
3. Conditions are `PipelineConfig` instances executed by the served pipeline,
   not a parallel implementation (eval/conditions.py).
4. The grid contains a genuinely unenriched control and a retrieval-free
   parametric baseline (eval/lexical.py).
5. Every reported difference carries a 95% CI from a paired bootstrap and a
   family-corrected p-value (eval/stats.py).
"""
