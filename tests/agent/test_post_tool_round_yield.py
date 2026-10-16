"""A trusted round-local yield ends the native turn without cancelling its work."""

import importlib.util
import json
import logging
from pathlib import Path
import threading
from unittest.mock import patch

import pytest


@pytest.fixture
def loop(monkeypatch, tmp_path):
    import hermes_cli.plugins as plugins
    import model_tools

    # Reuse the real-loop/SQLite harness rather than a second agent stand-in.
    spec = importlib.util.spec_from_file_location(
        "yield_persistence_support", Path(__file__).with_name("test_tool_call_incremental_persistence.py"))
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    owner = support._make_agent()
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kwargs: support._make_tool_defs("web_search", "process_manage"))
    db = support._attach_real_session_db(owner, tmp_path / "state.db", "yield-session")
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_tools.registry, "dispatch", lambda *args, **kwargs: json.dumps({"ok": True}))
    manager = plugins.PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    ctx = plugins.PluginContext(plugins.PluginManifest(name="yield-test"), manager)

    def run(hook, *, deferred=False, count=1, max_iterations=5, real_cleanup=False):
        registration = ctx.register_hook("post_tool_round", hook)
        owner.max_iterations = max_iterations
        tool_calls = [support._mock_tool_call("web_search", json.dumps({"query": str(index)}), f"call-{index}") for index in range(count)]
        if deferred:
            owner.valid_tool_names.add("tool_call")
            tool_calls = [support._mock_tool_call("tool_call", json.dumps({
                "calls": [{"name": "process_manage", "arguments": {"action": "poll", "session_id": "fixture-process"}}]}), "bridge-call")]
        owner.client.chat.completions.create.reset_mock()
        owner.client.chat.completions.create.side_effect = [
            support._mock_response("", "tool_calls", tool_calls), support._mock_response("normal final")]
        try:
            with patch.object(owner, "_save_trajectory"), patch.object(
                owner, "_cleanup_task_resources", wraps=owner._cleanup_task_resources if real_cleanup else None
            ) as cleanup:
                result = owner.run_conversation("do the task", task_id="owned-task")
                cleanup.assert_called_once_with("owned-task")
                return result
        finally:
            registration.dispose()

    yield owner, db, run, support, ctx
    db.close()


@pytest.mark.parametrize("deferred", [False, True])
def test_yield_follows_durable_results_and_never_requests_another_model_turn(loop, deferred):
    owner, db, run, support, ctx = loop
    durable_at_hook = []
    identities = []

    def hook(**kwargs):
        identities.append(kwargs)
        durable_at_hook.extend(db.get_messages_as_conversation(owner.session_id))
        return {"action": "yield"}

    result = run(hook, deferred=deferred, max_iterations=1)
    expected_id = "bridge-call" if deferred else "call-0"
    assert owner.client.chat.completions.create.call_count == 1
    assert [row["tool_call_id"] for row in durable_at_hook if row["role"] == "tool"] == [expected_id]
    assert identities[0]["tool_call_ids"] == (expected_id,)
    assert identities[0]["task_id"] == "owned-task"
    assert identities[0]["session_id"] == owner.session_id
    assert identities[0]["turn_id"] == result["turn_id"]
    assert set(identities[0]) == {"session_id", "task_id", "turn_id", "api_request_id", "tool_call_ids", "telemetry_schema_version"}
    assert json.loads(next(row["content"] for row in durable_at_hook if row["role"] == "tool"))["ok"] is True
    assert result["final_response"] == "[SILENT]"
    assert result["completed"] is True
    assert result["interrupted"] is False
    assert owner._interrupt_requested is False
    assert db.get_messages_as_conversation(owner.session_id)[-1]["content"] == "[SILENT]"


def test_yield_waits_for_whole_batch_and_keeps_output_hooks_without_generated_footer(loop, monkeypatch):
    owner, db, run, support, ctx = loop
    seen = []
    output = []
    monkeypatch.setattr("agent.turn_finalizer._append_file_mutation_footer", lambda *args: "wrong footer")
    monkeypatch.setattr("agent.turn_finalizer._explain_abnormal_exit", lambda *args: "wrong explanation")
    registration = ctx.register_hook("transform_llm_output", lambda text=None, **kwargs: output.append(kwargs) or None)

    def hook(**kwargs):
        seen.extend(row["tool_call_id"] for row in db.get_messages_as_conversation(owner.session_id) if row["role"] == "tool")
        return {"action": "yield"}

    try:
        result = run(hook, count=2)
    finally:
        registration.dispose()
    assert seen == ["call-0", "call-1"]
    assert result["final_response"] == "[SILENT]"
    assert output


@pytest.mark.parametrize("directive", [None, "yield", {"action": "yield", "extra": True}, {"action": "continue"}])
def test_only_exact_round_directive_yields(loop, directive):
    owner, db, run, support, ctx = loop
    result = run(lambda **kwargs: directive)
    assert owner.client.chat.completions.create.call_count == 2
    assert result["final_response"] == "normal final"


def test_timed_out_return_cannot_yield_this_or_the_next_turn(loop, monkeypatch):
    import hermes_cli.plugins as plugins

    owner, db, run, support, ctx = loop
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: 0.01)

    def hook(**kwargs):
        entered.set()
        release.wait(3)
        returned.set()
        return {"action": "yield"}

    try:
        result = run(hook)
        assert entered.is_set()
        assert result["final_response"] == "normal final"
    finally:
        release.set()
    assert returned.wait(3)
    next_result = run(lambda **kwargs: None)
    assert next_result["final_response"] == "normal final"
    assert owner.client.chat.completions.create.call_count == 2


@pytest.mark.parametrize("failure", ["persistence", "guardrail", "interrupt"])
def test_existing_failure_or_interrupt_wins_over_yield(loop, monkeypatch, failure):
    from types import SimpleNamespace

    owner, db, run, support, ctx = loop
    seen = []
    original = owner._execute_tool_calls

    def execute(*args, **kwargs):
        original(*args, **kwargs)
        if failure == "persistence":
            owner._incremental_persistence_failed = True
        elif failure == "guardrail":
            owner._tool_guardrail_halt_decision = SimpleNamespace(
                tool_name="web_search", code="fixture", to_metadata=lambda: {})
        else:
            owner._interrupt_requested = True

    monkeypatch.setattr(owner, "_execute_tool_calls", execute)
    monkeypatch.setattr(owner, "_toolguard_controlled_halt_response", lambda decision: "guardrail stopped")
    result = run(lambda **kwargs: seen.append(kwargs) or {"action": "yield"})
    assert not seen
    assert result["final_response"] != "[SILENT]"
    if failure != "guardrail":
        assert result["completed"] is False
    else:
        assert result["final_response"] == "guardrail stopped"
    if failure == "interrupt":
        assert result["interrupted"] is True


def test_interrupt_arriving_during_hook_is_not_reclassified_as_yield(loop):
    owner, db, run, support, ctx = loop

    def hook(**kwargs):
        owner._interrupt_requested = True
        return {"action": "yield"}

    result = run(hook)
    assert result["interrupted"] is True
    assert result["completed"] is False
    assert result["final_response"] != "[SILENT]"


def test_profile_switches_and_copied_context_do_not_inherit_a_yield(loop, tmp_path):
    from contextvars import copy_context
    import hermes_cli.plugins as plugins
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override, get_hermes_home

    owner, db, run, support, ctx = loop
    profiles = [tmp_path / "a", tmp_path / "b", tmp_path / "a"]
    events = []
    copied = []
    for index, home in enumerate(profiles):
        home.mkdir(exist_ok=True)
        token = set_hermes_home_override(home)
        try:
            manager = plugins.get_plugin_manager()
            manager._discovered = True
            scoped = plugins.PluginContext(plugins.PluginManifest(name="profile-yield"), manager)

            def hook(**kwargs):
                events.append((get_hermes_home(), kwargs["task_id"], kwargs["turn_id"]))
                copied.append(copy_context())
                return {"action": "yield"} if get_hermes_home() == profiles[0] else None

            registration = scoped.register_hook("post_tool_round", hook)
            try:
                result = run(lambda **kwargs: None)
            finally:
                registration.dispose()
            assert result["final_response"] == ("[SILENT]" if index != 1 else "normal final")
        finally:
            reset_hermes_home_override(token)
    assert [row[0] for row in events] == profiles
    assert {row[1] for row in events} == {"owned-task"}
    assert len({row[2] for row in events}) == 3
    # A context copied inside a yielding callback contains the profile, not a yield flag.
    result = copied[0].run(run, lambda **kwargs: None)
    assert result["final_response"] == "normal final"


def test_hook_exception_continues_normally_and_standalone_dispatch_cannot_yield(loop):
    import model_tools

    owner, db, run, support, ctx = loop
    seen = []

    def hook(**kwargs):
        seen.append(kwargs)
        raise RuntimeError("fixture hook failure")

    result = run(hook)
    assert result["final_response"] == "normal final"
    assert len(seen) == 1
    registration = ctx.register_hook("post_tool_round", hook)
    try:
        assert json.loads(model_tools.handle_function_call("web_search", {}))["ok"] is True
    finally:
        registration.dispose()
    assert len(seen) == 1


def test_yield_preserves_launched_and_unrelated_native_child_processes(loop, monkeypatch, tmp_path):
    import subprocess
    import sys
    import model_tools
    from tools.process_registry import ProcessRegistry

    owner, db, run, support, ctx = loop
    registry = ProcessRegistry()
    # Keep the real normal-finalizer VM cleanup; only the external browser is isolated.
    monkeypatch.setattr("run_agent.cleanup_browser", lambda task_id: None)
    script = tmp_path / "bounded_child.py"
    script.write_text("import time\ntime.sleep(30)\n")
    children = []
    sessions = []

    def launch(task):
        child = subprocess.Popen([sys.executable, str(script)], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        children.append(child)
        session = registry.adopt_local(child, command=str(script), cwd=str(tmp_path), task_id=task,
                                       owner_task_id=task, notify_on_complete=False)
        sessions.append(session)
        return json.dumps({"session_id": session.id, "status": "running", "exit_code": 0})

    try:
        launch("other-task")
        monkeypatch.setattr(model_tools.registry, "dispatch", lambda *args, **kwargs: launch("owned-task"))
        result = run(lambda **kwargs: {"action": "yield"}, real_cleanup=True)
        assert result["final_response"] == "[SILENT]"
        assert len(children) == 2
        assert all(child.poll() is None for child in children)
        assert all(registry.poll(session.id)["status"] == "running" for session in sessions)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)


@pytest.mark.parametrize("reason", ["yielded", "budget_exhausted"])
def test_yield_does_not_record_a_kanban_budget_failure(loop, monkeypatch, reason):
    import agent.turn_finalizer as finalizer

    owner, db, run, support, ctx = loop
    owner.max_iterations = 1
    recorded = []
    monkeypatch.setenv("HERMES_KANBAN_TASK", "fixture-task")
    monkeypatch.setattr(finalizer, "is_dispatcher_owned_worker_context", lambda: True)
    monkeypatch.setattr(finalizer, "_record_kanban_budget_exhausted", lambda *args: recorded.append(args))
    response = "[SILENT]" if reason == "yielded" else "bounded fallback"
    result = finalizer._resolve_budget_fallback(
        owner, final_response=response, api_call_count=1, interrupted=False, failed=False,
        messages=[], _turn_exit_reason=reason, _pending_verification_response=None,
        _pending_verification_response_previewed=False, logger=logging.getLogger(__name__))
    assert result[:2] == (response, reason)
    assert len(recorded) == (0 if reason == "yielded" else 1)
