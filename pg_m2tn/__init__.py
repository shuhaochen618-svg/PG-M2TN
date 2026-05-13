"""
PG-M2TN: Physics-Guided Masked Multi-Task Network for Edge Battery Diagnostics
================================================================================
A lightweight, edge-deployable framework for concurrent State of Health (SOH)
estimation and Voltage Distortion Ratio (VDR) diagnostics in lithium-ion batteries.

Paper: "Bridging Microscopic Polarization and Macroscopic Degradation:
        A Physics-Guided Masked Multi-Task Network for Edge Battery Diagnostics"

Repository: https://github.com/shuhaochen618-svg/PG-M2TN
"""

__version__ = "1.0.0"
__author__ = "Shuhao Chen"

from pg_m2tn.models.pg_m2tn import PGM2TN, count_parameters
from pg_m2tn.models.loss import PhysicsGatedLoss
from pg_m2tn.models.physics_extractor import PhysicsExtractor

__all__ = [
    "PGM2TN",
    "PhysicsGatedLoss",
    "PhysicsExtractor",
    "count_parameters",
]
