# Backend/app/models/__init__.py
# ---------------------------
# Package-level initializer for the `models` package.
# This file:
# 1. Marks `models/` as a Python package via the __init__.py.
# 2. Aggregates and re-exports all model classes from submodules,
#    so the rest of the app can do:
#       from Backend.app.models import DataGroup, ModelRun, Comment
#    instead of importing each submodule separately.
# 3. Provides an explicit __all__ list to control what’s public.
# ---------------------------

# Import domain-model groupings
from .data_models     import DataGroup, RealEstateData, LabourMarketData, BondYieldsData
from .run_models      import ModelRunConfig, ModelRun, ModelLossHistory
from .latent_models   import LatentPoint, LatentInput, PCAProjection, PCAProjectionConfig, SOMProjection, SOMProjectionConfig
from .feedback_models import Comment

# Define the public API of this package
__all__ = [
    # Data grouping & raw data tables
    "DataGroup", "RealEstateData", "LabourMarketData", "BondYieldsData",
    # Model run configuration and history
    "ModelRunConfig", "ModelRun", "ModelLossHistory",
    # Latent-space tables & configs
    "LatentPoint", "LatentInput", "PCAProjection", "PCAProjectionConfig", "SOMProjection", "SOMProjectionConfig",
    # Feedback/comments
    "Comment",
]
