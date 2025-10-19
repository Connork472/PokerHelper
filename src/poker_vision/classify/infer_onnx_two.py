"""Utility helpers for running the rank and suit ONNX models.

The original training pipeline for the lightweight CNNs renders 64×64
grayscale glyph tiles of the rank and suit symbols.  For runtime accuracy we
need to reproduce that preprocessing as closely as possible.  This module
therefore encapsulates:

* Corner extraction helpers to crop the card corner glyphs.
* Grayscale conversion, optional adaptive normalisation, and glyph centring
  that mirrors the synthetic training data generator.
* Convenience accessors that expose confidences, probability margins and the
  raw tiles for diagnostics/logging.

The :class:`TwoHead` class is the only public entry point.  It loads the two
ONNX models (rank and suit) and exposes :meth:`predict`,
:meth:`predict_with_conf`, and :meth:`predict_card` (full detail) helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


_RANK_LABELS = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"]
_SUIT_LABELS = ["s", "h", "d", "c"]


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax for a 2-D ``(batch, classes)`` array."""

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _resolve_margin(probabilities: np.ndarray, index: int) -> float:
    """Return the probability gap between the top-1 and runner-up classes."""

    if probabilities.shape[1] < 2:
        return 0.0
    ordered = np.sort(probabilities, axis=1)
    top = ordered[:, -1]
    runner_up = ordered[:, -2]
    return float(top[0] - runner_up[0])


@dataclass
class PredictionResult:
    """Container describing a single-card inference result."""

    label: str
    confidence: float
    margin: float
    rank_label: str
    suit_label: str
    rank_confidence: float
    suit_confidence: float
    rank_margin: float
    suit_margin: float
    corner: str
    processed_tile: np.ndarray

    def as_dict(self) -> Dict[str, float]:
        """Serialise the most useful scalar values for logging/debugging."""

        return {
            "label": self.label,
            "confidence": self.confidence,
            "margin": self.margin,
            "rank_label": self.rank_label,
            "rank_confidence": self.rank_confidence,
            "rank_margin": self.rank_margin,
            "suit_label": self.suit_label,
            "suit_confidence": self.suit_confidence,
            "suit_margin": self.suit_margin,
            "corner": self.corner,
        }


class TwoHead:
    """Thin wrapper around the rank + suit ONNX models.

    Parameters
    ----------
    rank_path, suit_path:
        Paths to the exported ONNX models.  Relative paths are resolved by the
        caller.
    providers:
        Optional ONNX Runtime execution providers.
    enable_adaptive_normalisation:
        When ``True`` (default) an adaptive CLAHE equaliser is applied prior to
        resizing, which stabilises predictions under varied lighting.  Set to
        ``False`` to use only min-max scaling (matching the baseline synthetic
        data).
    """

    def __init__(
        self,
        rank_path: str = "rank.onnx",
        suit_path: str = "suit.onnx",
        providers: Optional[List[str]] = None,
        *,
        enable_adaptive_normalisation: bool = True,
    ) -> None:
        providers = providers or ["CPUExecutionProvider"]
        self.rank = ort.InferenceSession(rank_path, providers=providers)
        self.suit = ort.InferenceSession(suit_path, providers=providers)
        self.height = int(self.rank.get_inputs()[0].shape[2])
        self.width = int(self.rank.get_inputs()[0].shape[3])
        self.rank_labels = _RANK_LABELS
        self.suit_labels = _SUIT_LABELS
        self.enable_adaptive_normalisation = enable_adaptive_normalisation
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    # ------------------------------------------------------------------
    # Glyph extraction + preprocessing helpers
    # ------------------------------------------------------------------
    def _extract_corner_tiles(self, card_bgr: np.ndarray) -> Iterable[Tuple[str, np.ndarray]]:
        """Yield the most informative card corner crops.

        The synthetic training set renders glyphs centred on a 64×64 canvas.
        Cropping too close to the card border therefore hurts accuracy.  These
        ratios were tuned empirically to capture the physical corner glyphs
        while discarding suit/body clutter.
        """

        if card_bgr.size == 0:
            return []

        h, w = card_bgr.shape[:2]
        pad_h = int(0.02 * h)
        pad_w = int(0.02 * w)
        crop_top = int(0.05 * h)
        crop_bottom = int(0.40 * h)
        left_inner = int(0.30 * w)
        right_inner = int(0.70 * w)

        top = max(0, crop_top - pad_h)
        bottom = min(h, crop_bottom + pad_h)

        left_tile = card_bgr[top:bottom, pad_w:left_inner]
        right_tile = card_bgr[top:bottom, right_inner : w - pad_w]

        tiles = []
        if left_tile.size:
            tiles.append(("top_left", left_tile))
        if right_tile.size:
            tiles.append(("top_right", right_tile))

        # A centred crop occasionally recovers labels missed by tight corner
        # crops (e.g. due to glare).  It is optional but inexpensive.
        centre_left = int(0.35 * w)
        centre_right = int(0.65 * w)
        centre_tile = card_bgr[top:bottom, centre_left:centre_right]
        if centre_tile.size:
            tiles.append(("top_centre", centre_tile))

        return tiles

    def _normalise_glyph(self, tile_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (model_input, processed_tile) pair for a raw corner crop."""

        gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)

        if self.enable_adaptive_normalisation:
            gray = self._clahe.apply(gray)
        else:
            gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Locate the glyph so we can centre it similar to the training data.
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            pad = int(0.08 * max(w, h))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(gray.shape[1], x + w + pad)
            y1 = min(gray.shape[0], y + h + pad)
            glyph = gray[y0:y1, x0:x1]
        else:
            glyph = gray

        resized = cv2.resize(glyph, (self.width, self.height), interpolation=cv2.INTER_AREA)
        model_input = resized.astype(np.float32) / 255.0
        return model_input[None, None, :, :], resized

    # ------------------------------------------------------------------
    # Public prediction helpers
    # ------------------------------------------------------------------
    def _predict_logits(self, tile_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference for a single crop and return logits + processed tile."""

        model_input, processed_tile = self._normalise_glyph(tile_bgr)
        rank_logits = self.rank.run(None, {self.rank.get_inputs()[0].name: model_input})[0]
        suit_logits = self.suit.run(None, {self.suit.get_inputs()[0].name: model_input})[0]
        return rank_logits, suit_logits, processed_tile

    def _build_prediction(self, corner: str, tile_bgr: np.ndarray) -> Optional[PredictionResult]:
        if tile_bgr.size == 0:
            return None

        rank_logits, suit_logits, processed_tile = self._predict_logits(tile_bgr)
        rank_probs = _softmax(rank_logits)
        suit_probs = _softmax(suit_logits)

        rank_index = int(np.argmax(rank_probs, axis=1)[0])
        suit_index = int(np.argmax(suit_probs, axis=1)[0])
        rank_conf = float(rank_probs[0, rank_index])
        suit_conf = float(suit_probs[0, suit_index])

        # Combine rank + suit confidences via geometric mean for stability.
        confidence = float(np.sqrt(rank_conf * suit_conf))

        rank_margin = _resolve_margin(rank_probs, rank_index)
        suit_margin = _resolve_margin(suit_probs, suit_index)
        margin = min(rank_margin, suit_margin)

        label = f"{self.rank_labels[rank_index]}{self.suit_labels[suit_index]}"

        return PredictionResult(
            label=label,
            confidence=confidence,
            margin=margin,
            rank_label=self.rank_labels[rank_index],
            suit_label=self.suit_labels[suit_index],
            rank_confidence=rank_conf,
            suit_confidence=suit_conf,
            rank_margin=rank_margin,
            suit_margin=suit_margin,
            corner=corner,
            processed_tile=processed_tile,
        )

    def predict_card(self, card_bgr: np.ndarray) -> PredictionResult:
        """Return the most confident prediction for the given card image."""

        best: Optional[PredictionResult] = None
        for corner, tile in self._extract_corner_tiles(card_bgr):
            result = self._build_prediction(corner, tile)
            if result is None:
                continue
            if best is None or result.confidence > best.confidence:
                best = result

        if best is None:
            return PredictionResult(
                label="",
                confidence=0.0,
                margin=0.0,
                rank_label="",
                suit_label="",
                rank_confidence=0.0,
                suit_confidence=0.0,
                rank_margin=0.0,
                suit_margin=0.0,
                corner="",
                processed_tile=np.zeros((self.height, self.width), dtype=np.uint8),
            )

        return best

    # Backwards-compatible helpers -------------------------------------------------
    def predict(self, card_bgr: np.ndarray) -> str:
        """Return the most likely label without exposing confidences."""

        return self.predict_card(card_bgr).label

    def predict_with_conf(self, card_bgr: np.ndarray, confidence_threshold: float = 0.55) -> Tuple[str, float]:
        """Return ``(label, confidence)`` applying ``confidence_threshold``."""

        result = self.predict_card(card_bgr)
        if result.confidence < confidence_threshold:
            return "", 0.0
        return result.label, result.confidence

