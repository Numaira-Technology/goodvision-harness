"""Offline measurement helpers for the Qwen3-VL-8B vLLM latency investigation.

These helpers do not rent a GPU, do not upload to R2, and do not change the
formal benchmark's requests, scores, or latency definition.
"""
