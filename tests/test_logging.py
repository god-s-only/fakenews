"""Tests for the logging configuration module."""

import logging
import os
from unittest import mock

from app.logging_config import configure_logging


def test_configure_logging_calls_basicconfig():
    with mock.patch("app.logging_config.logging.basicConfig") as mock_bc:
        configure_logging()
        mock_bc.assert_called_once()


def test_configure_logging_uses_env_log_level():
    with mock.patch.dict(os.environ, {"LOG_LEVEL": "WARNING"}):
        with mock.patch("app.logging_config.logging.basicConfig") as mock_bc:
            configure_logging()
            call_kwargs = mock_bc.call_args
            assert call_kwargs[1]["level"] == logging.WARNING


def test_configure_logging_defaults_to_info():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("LOG_LEVEL", None)
        with mock.patch("app.logging_config.logging.basicConfig") as mock_bc:
            configure_logging()
            call_kwargs = mock_bc.call_args
            assert call_kwargs[1]["level"] == logging.INFO


def test_configure_logging_handles_invalid_level():
    with mock.patch.dict(os.environ, {"LOG_LEVEL": "GARBAGE"}):
        with mock.patch("app.logging_config.logging.basicConfig") as mock_bc:
            configure_logging()
            call_kwargs = mock_bc.call_args
            assert call_kwargs[1]["level"] == logging.INFO
