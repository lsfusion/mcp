"""Client identity on events — and the honesty constraint that shapes it.

The model is not knowable from MCP by any supported route, so nothing here may
produce a model field the protocol did not carry. What these tests pin is that
the client IS recorded, that a caller-declared model is labelled as declared,
and that identity can never break a call.
"""
from __future__ import annotations

import pytest

from tools.caller import caller_fields, resolve_via


def test_records_the_client_and_its_user_agent(caller_context):
    caller_context(headers={"user-agent": "Cursor/3.19.13 (win32 x64)"})
    f = caller_fields()
    assert f["client"] == "Claude Code" and f["client_version"] == "2.1.263"
    assert f["user_agent"].startswith("Cursor/3.19.13")


def test_never_invents_a_model(caller_context):
    # The whole point: no supported MCP route carries the model, so the field
    # must be ABSENT rather than guessed from the client or the user agent.
    caller_context(headers={"user-agent": "Claude Code/2.1.263"})
    f = caller_fields()
    assert "model" not in f and "declared_model" not in f


def test_a_declared_model_is_labelled_as_declared(caller_context):
    # Nothing sends this today. If a harness ever does, the name must keep
    # saying that a caller asserted it and the protocol did not.
    caller_context(headers={"x-agent-model": "claude-opus-5"})
    assert caller_fields()["declared_model"] == "claude-opus-5"


def test_missing_request_or_session_degrades_to_partial(caller_context):
    caller_context(request=False)                      # stdio transport: no HTTP request
    assert caller_fields() == {"via": "dev", "client": "Claude Code", "client_version": "2.1.263"}
    caller_context(session=False, headers={"user-agent": "x"})
    assert caller_fields() == {"via": "dev", "user_agent": "x"}


def test_no_active_request_defaults_to_direct():
    # Nothing set: the default transport state outside a request.
    assert caller_fields() == {"via": "dev"}


def test_identity_can_never_break_a_call(monkeypatch):
    import mcp.server.lowlevel.server as srv

    class _Exploding:
        def get(self, default=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(srv, "request_ctx", _Exploding())
    assert caller_fields() == {"via": "dev"}


def test_values_are_capped(caller_context):
    caller_context(name="x" * 500, version="y" * 500,
                   headers={"user-agent": "u" * 500, "x-agent-model": "m" * 500})
    f = caller_fields()
    assert len(f["client"]) <= 60 and len(f["client_version"]) <= 40
    assert len(f["user_agent"]) <= 80 and len(f["declared_model"]) <= 60


@pytest.mark.parametrize("via", ["dev", "idedev", "app", "webapp"])
@pytest.mark.parametrize("name", ["lsfusion-intellij", "lsfusion-server", "Claude Code"])
def test_query_via_takes_precedence_over_client_fallback(caller_context, via, name):
    caller_context(name=name, via=via)
    assert caller_fields()["via"] == via
    assert "via_given" not in caller_fields()


@pytest.mark.parametrize("name,kind", [
    ("lsfusion-intellij", "idedev"),
    ("lsfusion-server", "legacy-server"),
    ("Claude Code", "dev"),
    ("unknown", "dev"),
])
def test_absent_query_falls_back_by_our_proxies_client_names(caller_context, name, kind):
    caller_context(name=name)
    assert caller_fields()["via"] == kind
    assert "via_given" not in caller_fields()


@pytest.mark.parametrize("value", ["unknown", "", "IDE", "ide", "webmcp", "skills", "skillsdev", "direct", "legacy-server", "x" * 500])
def test_unknown_query_is_direct_and_logs_capped_raw_value(caller_context, value):
    caller_context(name="lsfusion-server", via=value)
    fields = caller_fields()
    assert fields["via"] == "dev"
    assert fields["via_given"] == value[:60]
    assert resolve_via().articles == ("via_dev",)


@pytest.mark.parametrize("name", ["lsfusion-intellij", "lsfusion-server"])
def test_no_http_request_is_direct_even_for_a_proxy(caller_context, name):
    caller_context(name=name, request=False)
    assert caller_fields()["via"] == "dev"


def test_query_is_resolved_without_session_identity(caller_context):
    caller_context(via="webapp", session=False)
    assert caller_fields()["via"] == "webapp"


def test_via_changes_per_request_within_one_shared_proxy_session(caller_context):
    first = caller_context(name="lsfusion-server", via="app")
    assert resolve_via().articles == ("via_app",)
    second = caller_context(name="lsfusion-server", via="webapp")
    second.session = first.session
    assert caller_fields()["via"] == "webapp"
    assert resolve_via().articles == ("via_app", "via_webapp")
    third = caller_context(name="lsfusion-server")
    third.session = first.session
    assert caller_fields()["via"] == "legacy-server"


@pytest.mark.parametrize("via", [None, "dev", "idedev", "app", "webapp"])
def test_the_with_parameter_is_not_read(caller_context, via):
    # the skills plugin sends `with=skills`; the server reads nothing of it yet
    caller_context(via=via, addons=("skills",))
    with_param = (resolve_via(), caller_fields())
    caller_context(via=via)
    assert with_param == (resolve_via(), caller_fields())
