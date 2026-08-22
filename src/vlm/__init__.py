"""Vision-Language Model (VLM) module for edge sample usefulness evaluation."""

from .scorer import SampleScore, VLMScorer

__all__ = ["SampleScore", "VLMScorer"]
