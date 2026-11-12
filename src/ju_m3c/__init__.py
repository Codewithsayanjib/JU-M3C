"""JU-M3C: a large-scale, leakage-audited multimodal multi-layer network corpus of medical cases.

Importing this package pulls in nothing heavy. The reusable leakage audit lives in ``ju_m3c.audit``
(standard library only); the benchmark utilities in ``ju_m3c.benchmark`` import their own
dependencies lazily when used.
"""
__version__ = "0.1.0"
