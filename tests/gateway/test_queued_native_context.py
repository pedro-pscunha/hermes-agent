"""Queued turns bind their own native identity before launching more work."""

import asyncio
from contextvars import Context
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import _reply_anchor_for_event
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource, SessionStore
from gateway.session_context import capture_injection_origin, capture_injection_target, get_session_env
from tools.process_registry import ProcessRegistry
from tools.terminal_tool_background import _stamp_gateway_routing


@pytest.fixture
def queued(tmp_path, monkeypatch):
    support_spec = importlib.util.spec_from_file_location(
        "queued_routing_support", Path(__file__).with_name("test_background_task_routing.py"))
    support = importlib.util.module_from_spec(support_spec)
    support_spec.loader.exec_module(support)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(group_sessions_per_user=False)
    store = SessionStore(tmp_path / "sessions", config)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group",
                           user_id="42", user_name="A", message_id="message-A")
    entry = store.get_or_create_session(source)
    adapter = support.RoutingAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._MAX_INTERRUPT_DEPTH = 8
    monkeypatch.setattr(runner, "_run_agent_deliver_first_response", AsyncMock())
    monkeypatch.setattr(runner, "_prepare_profile_scoped_inbound_message_text", AsyncMock(return_value="queued"))
    monkeypatch.setattr(runner, "_refresh_agent_cache_message_count", AsyncMock())
    monkeypatch.setattr(runner, "_is_goal_continuation_event", lambda event: False)
    monkeypatch.setattr(adapter, "send_typing", AsyncMock())

    def bind(who, origin=None):
        runner._set_session_env(SessionContext(source=who, connected_platforms=[], home_channels={},
            session_key=entry.session_key, session_id=entry.session_id), injection_origin=origin)

    def turn(who, depth=0):
        return SimpleNamespace(source=who, session_id=entry.session_id, session_key=entry.session_key,
            run_generation=1, _interrupt_depth=depth, history=[], _status_thread_metadata=None,
            context_prompt="", result_holder=[None])

    async def follow(who, event, depth=0):
        return await runner._run_agent_queued_followup(turn(who, depth), adapter=adapter,
            pending="queued", pending_event=event, response="", result={"interrupted": True, "messages": []},
            stream_task=None)

    bind(source)
    target = capture_injection_target(entry.session_key)
    yield runner, source, entry, target, bind, follow
    runner._clear_session_env([])
    store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("outer_user", ["42", "99"])
async def test_queued_turn_and_follow_on_process_keep_their_requester(queued, monkeypatch, internal, outer_user):
    runner, source, entry, target, bind, follow = queued
    outer = replace(source, user_id=outer_user, user_name="B", message_id="message-B")
    outer_origin = replace(target, source=outer)
    bind(outer, outer_origin)
    outer_target = capture_injection_target(entry.session_key)
    event = MessageEvent(text="queued", source=source, message_id="message-A", internal=internal)
    if internal:
        event._injection_target = target
    observed = []

    async def execute(message, context_prompt, history, actual_source, session_id, **kwargs):
        assert actual_source == source
        assert kwargs["event_message_id"] == "message-A"
        assert kwargs["inbound_message_id"] == "message-A"

        def capture_process():
            process = ProcessRegistry._new_session("harmless", "task-A", "task-A", entry.session_key, "/tmp")
            _stamp_gateway_routing(process, get_session_env)
            observed.append((capture_injection_target(), capture_injection_origin(), process))

        await asyncio.to_thread(capture_process)
        return {"final_response": "done", "messages": []}

    monkeypatch.setattr(runner, "_run_agent_inner", execute)
    await follow(outer, event)
    current, origin, process = observed[0]
    assert current == target
    assert origin == (target if internal else None)
    assert process.injection_target == target.to_dict()
    assert process.watcher_message_id == "message-A"
    assert process.watcher_user_id == "42"
    assert process.parent_session_id == entry.session_id
    assert capture_injection_target() == outer_target
    assert capture_injection_origin() == outer_origin
    assert get_session_env("HERMES_SESSION_MESSAGE_ID") == "message-B"
    completion = runner._build_process_completion_event(
        {"message_id": process.watcher_message_id, "injection_target": process.injection_target}, process, process.id)
    completion_event = MessageEvent(text="finished", source=source, message_id=completion["message_id"])
    assert _reply_anchor_for_event(completion_event) == "message-A"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["error", "cancel"])
async def test_queued_context_restores_parent_after_failure_or_cancellation(queued, monkeypatch, outcome):
    runner, source, entry, target, bind, follow = queued
    outer = replace(source, user_id="99", message_id="message-B")
    bind(outer)
    parent = capture_injection_target()
    entered, stopped = asyncio.Event(), asyncio.Event()
    event = MessageEvent(text="queued", source=source, message_id="message-A")

    async def execute(*args, **kwargs):
        try:
            entered.set()
            assert capture_injection_target() == target
            if outcome == "error":
                raise ValueError("controlled failure")
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(runner, "_run_agent_inner", execute)
    if outcome == "error":
        with pytest.raises(ValueError, match="controlled failure"):
            await follow(outer, event)
    else:
        task = asyncio.create_task(follow(outer, event))
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert stopped.is_set()
    assert capture_injection_target() == parent
    assert get_session_env("HERMES_SESSION_MESSAGE_ID") == "message-B"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["ok", "error", "cancel"])
async def test_nested_queued_context_restores_each_parent(queued, monkeypatch, outcome):
    from gateway.platforms.base import ProcessingOutcome
    from gateway.run_turn_followup_ack import _followup_cancel_outcome

    runner, source, entry, target, bind, follow = queued
    outer = replace(source, user_id="99", message_id="message-B")
    third = replace(source, user_id="77", message_id="message-C")
    bind(outer)
    before = capture_injection_target()
    seen = []
    task = asyncio.current_task()
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter._expected_cancelled_tasks = {task}

    async def execute(message, context_prompt, history, actual_source, session_id, **kwargs):
        seen.append(get_session_env("HERMES_SESSION_MESSAGE_ID"))
        assert capture_injection_target().source == actual_source
        assert asyncio.current_task() is task
        assert _followup_cancel_outcome(adapter) == ProcessingOutcome.CANCELLED
        if actual_source == source:
            try:
                await follow(source, MessageEvent(text="third", source=third, message_id="message-C"), depth=1)
            finally:
                assert capture_injection_target() == target
        elif outcome == "error":
            raise ValueError("nested failure")
        elif outcome == "cancel":
            raise asyncio.CancelledError()
        return {"final_response": "done", "messages": []}

    monkeypatch.setattr(runner, "_run_agent_inner", execute)
    event = MessageEvent(text="queued", source=source, message_id="message-A")
    if outcome == "ok":
        await follow(outer, event)
    else:
        with pytest.raises(ValueError if outcome == "error" else asyncio.CancelledError):
            await follow(outer, event)
    assert seen == ["message-A", "message-C"]
    assert capture_injection_target() == before


@pytest.mark.parametrize("unset", [False, True])
@pytest.mark.parametrize("outcome", ["ok", "error", "cancel"])
def test_preservation_restores_exact_owned_values_and_cwd(tmp_path, unset, outcome):
    from agent import runtime_cwd
    from gateway import session_context as context

    child_dir = tmp_path / "child"
    child_dir.mkdir()

    def exercise():
        if not unset:
            context.set_session_vars(user_id="outer", message_id="outer-message", cwd=str(tmp_path),
                async_delivery=False, session_history_delivery="1")
        variables = (*context._VAR_MAP.values(), context._NATIVE_TARGET, context._INJECTION_ORIGIN,
                     context._SESSION_ASYNC_DELIVERY, context._SESSION_HISTORY_DELIVERY)
        previous = [(var, var.get()) for var in variables]
        previous_cwd = runtime_cwd._SESSION_CWD.get()

        def inner():
            with context.preserve_session_context():
                context.set_session_vars(user_id="inner", message_id="inner-message", cwd=str(child_dir),
                    async_delivery=True, session_history_delivery=None)
                assert runtime_cwd.resolve_agent_cwd() == child_dir
                if outcome == "error":
                    raise ValueError("restore")
                if outcome == "cancel":
                    raise asyncio.CancelledError()

        if outcome == "ok":
            inner()
        else:
            with pytest.raises(ValueError if outcome == "error" else asyncio.CancelledError):
                inner()
        assert all(var.get() is value for var, value in previous)
        assert runtime_cwd._SESSION_CWD.get() is previous_cwd

    Context().run(exercise)
