"""Shared token-state identifiers for feature-aware modality inputs.

The values are part of the cached-batch/model contract.  In particular,
feature masking changes only ``values`` and ``state_ids``: feature identity is
retained, while padding and an unavailable whole modality remain distinct.
"""

from __future__ import annotations

from enum import IntEnum


class TokenState(IntEnum):
    """Semantic state attached to each feature token."""

    OBSERVED = 0
    MASKED_VALUE = 1
    PADDING = 2
    MISSING_MODALITY = 3


OBSERVED_STATE = int(TokenState.OBSERVED)
MASKED_VALUE_STATE = int(TokenState.MASKED_VALUE)
PADDING_STATE = int(TokenState.PADDING)
MISSING_MODALITY_STATE = int(TokenState.MISSING_MODALITY)


__all__ = [
    "MASKED_VALUE_STATE",
    "MISSING_MODALITY_STATE",
    "OBSERVED_STATE",
    "PADDING_STATE",
    "TokenState",
]
