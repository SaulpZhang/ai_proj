"""Utilities for converting code strings into reduced embedding vectors."""

from __future__ import annotations

from collections.abc import Sequence
import os

import numpy as np
import openai


class CodeEmbeddingError(RuntimeError):
	"""Raised when code embedding or PCA reduction fails."""


class CodeEmbedder:
	"""Embeds code strings and reduces vectors to a fixed PCA dimension."""

	def __init__(
			self,
			model: str = 'text-embedding-3-small',
			base_url: str | None = None,
			api_key: str | None = None,
			timeout: int = 120,
	) -> None:
		self._model = model
		resolved_api_key = api_key or os.getenv('OPENAI_API_KEY')
		if not resolved_api_key:
			raise ValueError(
					'Missing API key. Set `OPENAI_API_KEY` or pass `api_key`.')
		self._client = openai.OpenAI(
				base_url=base_url,
				api_key=resolved_api_key,
				timeout=timeout,
		)

	def embed_code(self, code: str) -> np.ndarray:
		"""Returns the raw embedding vector for one code string."""
		if not code or not code.strip():
			raise ValueError('`code` must be a non-empty string.')

		try:
			response = self._client.embeddings.create(
					model=self._model,
					input=code,
			)
			vector = np.array(response.data[0].embedding, dtype=np.float32)
		except Exception as exc:  # Network/SDK errors.
			raise CodeEmbeddingError(f'Failed to embed code: {exc}') from exc

		if vector.ndim != 1 or vector.size == 0:
			raise CodeEmbeddingError('Embedding response has invalid shape.')
		return vector

	@staticmethod
	def _reduce_with_pca(vectors: np.ndarray, out_dim: int) -> np.ndarray:
		"""Runs PCA using SVD and returns projected vectors."""
		if vectors.ndim != 2:
			raise ValueError('`vectors` must be a 2D array.')
		if out_dim <= 0:
			raise ValueError('`out_dim` must be positive.')

		num_samples, num_features = vectors.shape
		if out_dim > num_features:
			raise ValueError('`out_dim` cannot exceed embedding dimension.')

		# For very small sample count, real PCA rank is limited by sample count.
		max_rank = min(num_samples - 1, num_features)
		if max_rank <= 0:
			# Single-sample fallback: deterministic truncate/pad.
			single = vectors[0]
			if out_dim <= single.size:
				return single[:out_dim][None, :]
			padded = np.pad(single, (0, out_dim - single.size))
			return padded[None, :]

		work_dim = min(out_dim, max_rank)
		centered = vectors - np.mean(vectors, axis=0, keepdims=True)
		_, _, vt = np.linalg.svd(centered, full_matrices=False)
		components = vt[:work_dim].T
		reduced = centered @ components

		# If work_dim < out_dim (happens with tiny batch), right-pad with zeros.
		if reduced.shape[1] < out_dim:
			reduced = np.pad(reduced, ((0, 0), (0, out_dim - reduced.shape[1])))
		return reduced.astype(np.float32, copy=False)

	def embed_and_reduce(
			self,
			code: str,
			pca_dim: int = 16,
			reference_codes: Sequence[str] | None = None,
	) -> tuple:
		"""Embeds `code` and returns a PCA-reduced vector of length `pca_dim`.

		Args:
			code: Target code string to encode.
			pca_dim: Output vector dimension after PCA.
			reference_codes: Optional additional code samples used to fit PCA basis.
					If omitted, a single-sample fallback is used.

		Returns:
			A 1D numpy vector with shape `(pca_dim,)`.
		"""
		target_vector = self.embed_code(code)

		vectors = [target_vector]
		if reference_codes:
			for ref_code in reference_codes:
				if ref_code and ref_code.strip():
					vectors.append(self.embed_code(ref_code))

		stacked = np.vstack(vectors)
		reduced = self._reduce_with_pca(stacked, out_dim=pca_dim)
		return tuple(reduced[0])


def embed_code_to_16d(
		code: str,
		reference_codes: Sequence[str] | None = None,
		model: str = 'text-embedding-3-small',
		base_url: str = "https://api.bltcy.ai/v1",
		api_key: str = "sk-KcOna1MllF2BWDRIeXpcHAY92TKvHox3KnfpXsBZHiydDD0n",
) -> tuple:
	"""Convenience helper: code string -> embedding -> 16D vector."""
	embedder = CodeEmbedder(
			model=model,
			base_url=base_url,
			api_key=api_key,
	)
	return embedder.embed_and_reduce(
			code=code,
			pca_dim=16,
			reference_codes=reference_codes,
	)
