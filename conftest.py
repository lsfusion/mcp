"""Session setup for the whole suite: what has to be in place before any test
module is imported, and the CI switch that makes a skipped test fail."""
from __future__ import annotations

import os

import pytest

# No test talks to OpenAI, but tools.rag_retrieve builds its client at import
# from settings.OPENAI_API_KEY, the client refuses an empty key, and settings
# reads the variable once, when it is first imported. A key set inside a test
# module came too late whenever another module had imported settings first, and
# the suite failed to collect. A root conftest is imported before any test
# module, so the key is in place for all of them. It is forced rather than
# defaulted: a developer's real key must not be what a test that lost its stub
# reaches the API with.
os.environ["OPENAI_API_KEY"] = "sk-test-dummy"


def pytest_addoption(parser):
    parser.addoption(
        "--forbid-skips", action="store_true",
        help="fail every test that would skip. Each skip in this suite means a "
             "sibling checkout (platform, plugin-idea) is missing, which CI "
             "provides, so there a skip is a check that silently did not run.")


def _forbidden(report, config) -> bool:
    # An expected failure is reported as skipped too, but it ran.
    return (report.skipped and not hasattr(report, "wasxfail")
            and config.getoption("forbid_skips", default=False))


def _as_failure(report):
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
    reason = reason.removeprefix("Skipped: ")
    report.outcome = "failed"
    report.longrepr = f"skipped, and --forbid-skips is on: {reason}"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if _forbidden(report, item.config):
        _as_failure(report)
    return report


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    report = yield
    if _forbidden(report, collector.config):
        _as_failure(report)
    return report
