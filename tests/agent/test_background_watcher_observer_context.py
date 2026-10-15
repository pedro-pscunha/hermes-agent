"""A dispatch's observer ownership must not leak into a background completion turn."""

import asyncio
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def observed(monkeypatch):
    import hermes_cli.lifecycle as lifecycle

    events = []
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "post_tool_call")
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda name, **fields:
                        (events.append(fields) if name == "post_tool_call" else None) or [])
    return events


def agent():
    return SimpleNamespace(
        session_id="owned-session", _current_turn_id="launch-turn", _current_api_request_id="request",
        _context_engine_tool_names=set(), _memory_manager=None, quiet_mode=False,
        valid_tool_names=set(), enabled_toolsets=None, disabled_toolsets=None,
    )


def managed_call(owner, name, args, call_id):
    from agent.tool_executor import _ToolCallRef, _resolve_sequential_dispatch

    ref = _ToolCallRef(name, args, owner.session_id, call_id, [])
    result = _resolve_sequential_dispatch(owner, ref, []).execute(args)
    # The same terminal publisher used after the sequential worker returns.
    ref.emit_post(owner, result)
    return result


def test_native_watcher_schedule_preserves_profile_without_muting_completion(monkeypatch, tmp_path, observed):
    import model_tools
    from gateway.run_notifications import GatewayNotificationsMixin
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    homes = [tmp_path / "dona", tmp_path / "other", tmp_path / "dona"]
    for home in set(homes):
        home.mkdir()
    profiles = []

    async def run():
        owner = agent()
        queue = asyncio.Queue()

        async def watcher(descriptor):
            # The production scheduling boundary has copied the launching worker context.
            await asyncio.sleep(0)
            profiles.append(get_hermes_home())
            model_tools.handle_function_call("read_file", {}, tool_call_id=descriptor["id"] + "-immediate",
                                             session_id="owned-session", turn_id="completion-turn")
            await queue.put(descriptor)

        runner = SimpleNamespace(_gateway_loop=asyncio.get_running_loop(), _running=True,
                                 _run_process_watcher=watcher)

        def dispatch(name, args, **kwargs):
            if name == "terminal":
                assert GatewayNotificationsMixin.arm_process_watcher(runner, args)
            return json.dumps({"status": "exited", "exit_code": 0})

        monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
        for index, home in enumerate(homes):
            def launch():
                token = set_hermes_home_override(home)
                try:
                    managed_call(owner, "terminal", {"id": str(index)}, str(index) + "-launch")
                finally:
                    reset_hermes_home_override(token)

            await asyncio.to_thread(launch)
            descriptor = await asyncio.wait_for(queue.get(), 5)
            # A queued completion runs in an independent consumer context.
            token = set_hermes_home_override(home)
            try:
                model_tools.handle_function_call("read_file", {}, tool_call_id=descriptor["id"] + "-queued",
                                                 session_id="owned-session", turn_id="queued-turn")
            finally:
                reset_hermes_home_override(token)

    asyncio.run(run())
    assert profiles == homes
    assert sorted(row["tool_call_id"] for row in observed) == sorted(
        f"{index}-{kind}" for index in range(3) for kind in ("launch", "immediate", "queued"))


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_direct_and_local_bridge_have_one_terminal_event(monkeypatch, observed, managed, raises):
    import model_tools

    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kwargs: [{
        "type": "function", "function": {"name": "process_manage", "description": "Poll",
        "parameters": {"type": "object", "properties": {}}}}])

    def dispatch(*args, **kwargs):
        if raises:
            raise RuntimeError("fixture failure")
        return json.dumps({"result": "receipt"})

    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    args = {"calls": [{"name": "process_manage", "arguments": {"action": "poll", "session_id": "fixture"}}]}
    if managed:
        result = managed_call(agent(), "tool_call", args, "bridge")
    else:
        result = model_tools.handle_function_call("tool_call", args, tool_call_id="bridge")
    assert ("error" in json.loads(result)) is raises
    assert len(observed) == 1
    assert observed[0]["tool_call_id"] == "bridge"
    # A failed invocation cannot poison the next independent direct call.
    model_tools.handle_function_call("read_file", {}, tool_call_id="next")
    assert [row["tool_call_id"] for row in observed] == ["bridge", "next"]


@pytest.mark.parametrize("managed", [False, True])
def test_real_connector_batch_preserves_entry_policy_and_observer_ownership(monkeypatch, observed, managed):
    import model_tools
    from tools.connectors.gateway import bridge, config
    from tools.registry import invalidate_check_fn_cache

    monkeypatch.setattr(config, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "connectors_available", lambda: True)
    invalidate_check_fn_cache()
    names = ["connectors__gmail__READ_EMAIL", "connectors__calendar__LIST_EVENTS"]
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kwargs: [{
        "type": "function", "function": {"name": "manage_connections", "parameters": {"type": "object"}}}])
    monkeypatch.setattr(model_tools, "_select_tool_names", lambda *args, **kwargs: ["manage_connections"])
    policies = []
    import hermes_cli.plugins as plugins
    monkeypatch.setattr(plugins, "_dispatch_pre_tool_call_hooks",
                        lambda name, args, **kwargs: (policies.append(name) or None, None))
    # Only the external network boundary is replaced; partitioning and recursive dispatch are real.
    monkeypatch.setattr(bridge, "run_remote", lambda plans, *args, **kwargs: [
        {"index": plan.position, "name": plan.name, "response": {"ok": True}} for plan in plans])
    args = {"calls": [{"name": name, "arguments": {}} for name in names]}
    if managed:
        result = managed_call(agent(), "tool_call", args, "batch")
    else:
        result = model_tools.handle_function_call("tool_call", args, tool_call_id="batch")
    assert "error" not in json.loads(result)
    assert policies == names
    assert [row["tool_name"] for row in observed] == (["tool_call"] if managed else names + ["tool_call"])


def test_nested_distinct_invocation_keeps_its_own_observer(monkeypatch, observed):
    import model_tools

    def dispatch(name, args, **kwargs):
        if name == "execute_code":
            return model_tools.handle_function_call("read_file", {}, tool_call_id="child")
        return json.dumps({"result": "receipt"})

    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    managed_call(agent(), "execute_code", {}, "parent")
    assert [row["tool_call_id"] for row in observed] == ["child", "parent"]
