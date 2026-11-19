"""Tests for settings.py — the build identifier the event log stamps, and where
reports are stored."""
from __future__ import annotations

import os

from settings import _reports_dir, _server_version


def test_the_commit_baked_into_the_image_is_the_version():
    assert _server_version({"MCP_BUILD_SHA": "70dfeb5aa"}) == "70dfeb5aa"


def test_a_build_without_a_commit_is_unknown():
    # A local `docker build` without --build-arg leaves MCP_BUILD_SHA empty.
    assert _server_version({"MCP_BUILD_SHA": ""}) == "unknown"
    assert _server_version({}) == "unknown"
    # The retired hand-set variable no longer names a build.
    assert _server_version({"MCP_SERVER_VERSION": "2da49f14c90c"}) == "unknown"


def test_reports_live_under_the_log_dir_unless_placed_elsewhere():
    # On ai LOG_DIR is the bind mount, so reports need no mount of their own.
    assert _reports_dir({"LOG_DIR": "/data/logs"}) == os.path.join("/data/logs", "reports")
    assert _reports_dir({"LOG_DIR": "/data/logs", "REPORTS_DIR": "/srv/rep"}) == "/srv/rep"
    assert _reports_dir({"REPORTS_DIR": "/srv/rep"}) == "/srv/rep"


def test_no_dir_at_all_means_nowhere_to_store():
    assert _reports_dir({}) == ""
    assert _reports_dir({"LOG_DIR": "", "REPORTS_DIR": ""}) == ""
