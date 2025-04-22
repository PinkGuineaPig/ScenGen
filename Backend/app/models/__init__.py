# Backend/app/models/__init__.py

from .run_models      import ModelRunConfig, ModelRun, ModelLossHistory
# …etc…

# You need to add Currency, ExchangeRate here too (if you still want them on the package-level API):
from .data_models     import (
    Currency, ExchangeRate,
)

from .feedback_models     import (
    Comment
)

from .latent_models     import (
    LatentPoint,LatentInput,SOMProjectionConfig,SOMProjection,PCAProjectionConfig,PCAProjection
)

__all__ = [
    "ModelRunConfig", "ModelRun", "ModelLossHistory",
    "LatentPoint", "LatentInput", "PCAProjection", "PCAProjectionConfig",
    "SOMProjection", "SOMProjectionConfig",
    "Comment",
    # ← and here:
    "Currency", "ExchangeRate",
]
