"""Compatibility names for integrations using the original AIClub client."""

from .model_api import (
    DEFAULT_AICLUB_API_KEY_ENV as DEFAULT_AICLUB_API_KEY_ENV,
    DEFAULT_AICLUB_ENDPOINT as DEFAULT_AICLUB_ENDPOINT,
    ModelAPIClient as AIClubClient,
    ModelAPIError as AIClubError,
    ModelAPISettings as AIClubSettings,
    api_key_present as api_key_present,
)
