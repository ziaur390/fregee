"""Drift detection.

PSI is the decision metric; KS is recorded only as corroboration. The reasoning
is in ``configs/drift.yaml``.

Import from the submodules directly, so that ``python -m mlserve.drift.detect``
does not warn about the package having already imported the module it is about to
run:

    from mlserve.drift.reference import build_reference, load_reference
    from mlserve.drift.detect import detect
"""
