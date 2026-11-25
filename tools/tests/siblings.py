"""Where the sibling repositories are, for the tests that read them.

Some checks read files of the repositories mcp is mirrored into: the drift guard
reads the platform and plugin proxies and the platform's guidance docs, and the
chunker test chunks real platform docs. Those tests carry the `siblings` mark.

By default the siblings sit next to mcp, as in the super-workspace, where the
plugin is checked out as `plugin_idea` (GitHub names it `plugin-idea`; either is
found). `LSF_PLATFORM_ROOT` / `LSF_PLUGIN_ROOT` point a check elsewhere, e.g. at
the worktree that carries a mirroring change. CI sets both.

One rule for every such read. A file missing under a root found by default
means the sibling is not checked out here, and the test skips. A file missing
under a root set explicitly is a typo or a moved file, and the test fails:
skipping it would turn the guard green while it checks nothing.
"""
from __future__ import annotations

import os
import pathlib

import pytest

# tools/tests/ -> tools -> mcp -> the directory that holds the siblings
_WORKSPACE = pathlib.Path(__file__).resolve().parents[3]
_VARS = ("LSF_PLATFORM_ROOT", "LSF_PLUGIN_ROOT")


def _root(var: str, *names: str) -> pathlib.Path:
    if os.environ.get(var):
        return pathlib.Path(os.environ[var])
    candidates = [_WORKSPACE / name for name in names]
    return next((p for p in candidates if p.is_dir()), candidates[0])


PLATFORM_ROOT = _root("LSF_PLATFORM_ROOT", "platform")
PLUGIN_ROOT = _root("LSF_PLUGIN_ROOT", "plugin-idea", "plugin_idea")
_EXPLICIT = {root: var for var, root in zip(_VARS, (PLATFORM_ROOT, PLUGIN_ROOT))
             if os.environ.get(var)}


def need(path: pathlib.Path) -> pathlib.Path:
    """`path` if it exists; otherwise skip the test, or fail it when the root
    `path` lies under was set explicitly."""
    if path.exists():
        return path
    for root, var in _EXPLICIT.items():
        if path.is_relative_to(root):
            pytest.fail(f"{var} is set, but {path} is missing", pytrace=False)
    pytest.skip(f"{path} is missing: no sibling checkout here? (see {' / '.join(_VARS)})")
