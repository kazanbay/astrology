"""Unit tests of the log privacy filter."""

import logging

from logging_utils import PrivacyFilter


def record(message, args=()):
    return logging.LogRecord("test", logging.INFO, __file__, 1, message, args, None)


def test_masks_email_and_coordinates():
    log_filter = PrivacyFilter()
    item = record("contact %s from %s wrote to a@b.co at 55.7558,37.6173", ("Сергей", "42"))

    assert log_filter.filter(item) is True
    message = item.getMessage()
    assert "a@b.co" not in message and "<email>" in message
    assert "55.7558" not in message and "<coord>" in message


def test_masks_long_numeric_ids():
    log_filter = PrivacyFilter()
    item = record("telegram_id=1234567890123 handled")
    log_filter.filter(item)

    message = item.getMessage()
    assert "1234567890123" not in message
    assert "<id>" in message


def test_keeps_hashes_and_short_numbers():
    log_filter = PrivacyFilter()
    digest = "a" * 64
    item = record(f"hash {digest}; orb 0.32°; user_id=7")
    log_filter.filter(item)

    message = item.getMessage()
    assert digest in message
    assert "0.32" in message
    assert "user_id=7" in message
