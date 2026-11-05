"""PCA baseline tokenizer — linear, non-trainable, fits via SVD on first batches."""

from ntt.contrib.tokenizer.pca_baseline.pca_baseline import (
    PCABaselineTokenizer,
    PCABaselineTokenizerSpec,
)

__all__ = ["PCABaselineTokenizer", "PCABaselineTokenizerSpec"]
