"""Tests for settings.py — which build identifier the event log stamps."""
from __future__ import annotations

from settings import _server_version


def test_commit_baked_into_the_image_wins_over_the_hand_set_value():
    # The deploy host's .env pins MCP_SERVER_VERSION; the image's commit must win.
    env = {"MCP_BUILD_SHA": "70dfeb5aa", "MCP_SERVER_VERSION": "2da49f14c90c"}
    assert _server_version(env) == "70dfeb5aa"


def test_a_build_without_a_commit_falls_back():
    # A local `docker build` without --build-arg leaves MCP_BUILD_SHA empty.
    assert _server_version({"MCP_BUILD_SHA": "", "MCP_SERVER_VERSION": "dev-1"}) == "dev-1"
    assert _server_version({}) == "unknown"
