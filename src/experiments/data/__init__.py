"""Data building: strong-label exceedance rules."""

from .labels import DEFAULT_RULES, ExceedanceRule, build_strong_labels, is_exceedance

__all__ = ["ExceedanceRule", "DEFAULT_RULES", "is_exceedance", "build_strong_labels"]
