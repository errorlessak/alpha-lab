"""Route modules, one per resource family in §14.1."""

from __future__ import annotations

from alphaservice.api.routers import datasets, jobs, runs, system

__all__ = ["datasets", "jobs", "runs", "system"]
