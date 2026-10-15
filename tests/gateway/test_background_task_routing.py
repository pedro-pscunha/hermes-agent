"""Task-owned wakes preserve native requester, route and compression lineage."""

import asyncio
import json
from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import BasePlatformAdapter, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource, SessionStore
from gateway.session_context import InjectionTarget, capture_injection_target, get_session_env
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_state import AsyncSessionDB, SessionDB
from tools.process_registry import ProcessRegistry, ProcessSession
from tools.terminal_tool_background import _register_completion_watcher, _stamp_gateway_routing


class RoutingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test", extra={"group_sessions_per_user": False}), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise AssertionError("no network delivery")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "group"}


@pytest.fixture
def route(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42,alt-42")
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"entries": {"task-owner": {"allow_gateway_injection": True}}},
    }))
    db = SessionDB(tmp_path / "state.db")
    config = GatewayConfig(group_sessions_per_user=False)
    store = SessionStore(tmp_path / "sessions", config)
    store._db = db
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group",
                           user_id="42", user_id_alt="alt-42", user_name="Original requester",
                           thread_id="7", scope_id="tenant-a", parent_chat_id="-100")
    entry = store.get_or_create_session(source)
    adapter = RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    adapter._active_sessions[entry.session_key] = asyncio.Event()
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.session_store = store
    runner._session_db = AsyncSessionDB(db)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._running = True
    runner._draining = False
    runner._background_tasks = set()
    runner._queued_events = {}
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_: False)
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    context = PluginContext(PluginManifest(name="task-owner", key="task-owner", source="user"), PluginManager())
    tokens = runner._set_session_env(SessionContext(source=source, connected_platforms=[],
        home_channels={}, session_key=entry.session_key, session_id=entry.session_id))
    target = context.capture_injection_target(entry.session_key)
    assert target is not None
    runner._clear_session_env(tokens)
    yield runner, adapter, entry, target, db, context
    runner._clear_session_env([])
    store.close_all_db_handles()
    db.close()


def test_capture_uses_bound_native_source_and_never_ambient_environment(route, monkeypatch):
    runner, _, entry, target, _, context = route
    monkeypatch.setenv("HERMES_SESSION_ID", "foreign")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "foreign")
    assert context.capture_injection_target(entry.session_key) is None
    source = replace(target.source)
    runner._set_session_env(SessionContext(source=source, connected_platforms=[], home_channels={},
        session_key=entry.session_key, session_id=entry.session_id))
    captured = context.capture_injection_target()
    assert captured == target
    source.user_id = "other"
    assert captured.source.user_id == "42"
    assert captured.source.user_id_alt == "alt-42"
    assert context.capture_injection_target("wrong-route") is None
    assert InjectionTarget.from_dict(json.loads(json.dumps(captured.to_dict()))) == captured


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["same", "compression", "reset", "unrelated-live", "revoked"])
async def test_pinned_dispatch_uses_original_requester_and_verified_lineage(route, monkeypatch, mode):
    runner, adapter, entry, target, db, _ = route
    entry.origin = replace(target.source, user_id="99", user_id_alt="alt-99", user_name="Later requester")
    if mode in {"compression", "reset", "unrelated-live"}:
        if mode != "unrelated-live":
            db.end_session(target.session_id, end_reason="compression" if mode == "compression" else "session_reset")
        db.create_session("next-session", source="telegram",
                          parent_session_id=target.session_id if mode == "compression" else None)
        entry.session_id = "next-session"
    if mode == "revoked":
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "99")
    accepted = await runner._dispatch_plugin_message_injection(
        session_key=target.session_key, content="task-owned nonce", plugin_id="task-owner", target=target)
    assert accepted is (mode in {"same", "compression"})
    if not accepted:
        assert not runner._queued_events
        return
    event = adapter._pending_messages[target.session_key]
    assert event.source == target.source
    assert event.metadata["gateway_session_id"] == entry.session_id
    assert event.metadata["gateway_session_strict"] is True
    tokens = runner._set_session_env(SessionContext(source=event.source, connected_platforms=[], home_channels={},
        session_key=entry.session_key, session_id=entry.session_id))
    try:
        assert get_session_env("HERMES_SESSION_USER_ID") == "42"
        assert get_session_env("HERMES_SESSION_USER_ID_ALT") == "alt-42"
        assert get_session_env("HERMES_SESSION_SCOPE_ID") == "tenant-a"
    finally:
        runner._clear_session_env(tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("profile", "foreign"), ("scope_id", "other-tenant"),
                                         ("thread_id", "other-thread"), ("chat_id", "other-chat")])
async def test_target_cannot_cross_route_discriminators(route, field, value):
    runner, _, _, target, _, _ = route
    foreign = replace(target, source=replace(target.source, **{field: value}))
    assert not await runner._dispatch_plugin_message_injection(
        session_key=target.session_key, content="wake", plugin_id="task-owner", target=foreign)
    assert not runner._queued_events


@pytest.mark.asyncio
async def test_dispatch_requires_real_adapter_acceptance(route):
    runner, adapter, _, target, _, _ = route
    adapter.handle_message = AsyncMock()
    assert not await runner._dispatch_plugin_message_injection(
        session_key=target.session_key, content="wake", plugin_id="task-owner", target=target)


@pytest.mark.asyncio
async def test_full_native_fifo_refuses_task_injection_without_replacing_pending(route):
    runner, adapter, _, target, _, _ = route
    runner._BUSY_QUEUE_MAX_PENDING = 1
    assert await runner._dispatch_plugin_message_injection(
        session_key=target.session_key, content="first", plugin_id="task-owner", target=target)
    assert not await runner._dispatch_plugin_message_injection(
        session_key=target.session_key, content="second", plugin_id="task-owner", target=target)
    assert adapter._pending_messages[target.session_key].text == "first"
    assert not runner._queued_events


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["plugin", "process"])
@pytest.mark.parametrize("change", ["unchanged", "allowlist", "pairing", "compression", "reset"])
async def test_queued_target_rechecks_authority_and_lineage_at_native_start(route, monkeypatch, producer, change):
    from gateway.pairing import PairingStore

    runner, adapter, entry, target, db, context = route
    if change == "pairing":
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS")
        runner.pairing_store = PairingStore()
        code = runner.pairing_store.generate_code("telegram", "42")
        assert runner.pairing_store.approve_code("telegram", code)
    if producer == "plugin":
        assert await runner._dispatch_plugin_message_injection(
            session_key=target.session_key, content="wake", plugin_id="task-owner", target=target)
    else:
        assert await runner._inject_watch_notification("wake", {
            "session_key": target.session_key, "injection_target": target.to_dict()})
    event = adapter._pending_messages.pop(target.session_key)
    if change == "allowlist":
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "99")
    elif change == "pairing":
        assert runner.pairing_store.revoke("telegram", "42")
    elif change in {"compression", "reset"}:
        db.end_session(target.session_id, end_reason="compression" if change == "compression" else "session_reset")
        db.create_session("after-queue", source="telegram",
                          parent_session_id=target.session_id if change == "compression" else None)
        entry.session_id = "after-queue"
    admitted = await runner._hm_admit_event(event)
    if change in {"allowlist", "pairing"}:
        assert admitted is None
        return
    assert admitted is not None
    resolved = await runner._hmwa_resolve_session(admitted[0], admitted[1])
    if change == "reset":
        assert resolved is None
    else:
        assert resolved[1].session_id == entry.session_id
        assert resolved[0].user_id == "42"
        tokens = runner._set_session_env(SessionContext(source=resolved[0], connected_platforms=[],
            home_channels={}, session_key=resolved[2], session_id=resolved[1].session_id),
            injection_origin=event._injection_target)
        assert context.capture_injection_origin() == target
        assert context.capture_injection_target().session_id == entry.session_id
        runner._clear_session_env(tokens)
        assert context.capture_injection_origin() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_allowed", [True, False])
async def test_durable_target_keeps_primary_transport_when_runtime_owns_another_bot(route, tmp_path, monkeypatch, primary_allowed):
    from gateway.run import _profile_runtime_scope
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    import hermes_cli.profiles as profiles
    import tools.process_registry as registry_module

    runner, primary, _, _, _, context = route
    home = tmp_path / "profiles" / "runtime"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n")
    (tmp_path / ".env").write_text(f"TELEGRAM_ALLOWED_USERS={'42' if primary_allowed else '99'}\n")
    (home / ".env").write_text(f"TELEGRAM_ALLOWED_USERS={'99' if primary_allowed else '42'}\n")
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: tmp_path / "profiles")
    monkeypatch.setattr(registry_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    secondary = RoutingAdapter()
    runner._profile_adapters = {"runtime": {Platform.TELEGRAM: secondary}}
    source = primary.build_source(chat_id="-100", chat_type="group", user_id="42", user_id_alt="alt-42")
    source.profile = "runtime"
    entry = runner.session_store.get_or_create_session(source)
    runner.config.multiplex_profiles = True
    tokens = runner._set_session_env(SessionContext(source=source, connected_platforms=[],
        home_channels={}, session_key=entry.session_key, session_id=entry.session_id))
    captured = context.capture_injection_target()
    assert captured.transport_profile is None
    assert captured.transport_platform == "telegram"
    registry = ProcessRegistry()
    process = registry._new_session(command="bounded-task", task_id="task", owner_task_id="task",
        session_key=entry.session_key, cwd=str(tmp_path), pid=4242, host_start_time=123.0, notify_on_complete=True)
    registry._running[process.id] = process
    _stamp_gateway_routing(process, get_session_env)
    _register_completion_watcher(registry, process, entry.session_key)
    runner._clear_session_env(tokens)
    restored = ProcessRegistry()
    monkeypatch.setattr(restored, "_host_pid_is_ours", lambda *_: True)
    assert restored.recover_from_checkpoint() == 1
    target = InjectionTarget.from_dict(restored.pending_watchers[0]["injection_target"])
    rebound = runner._source_for_injection_target(target)
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    try:
        with _profile_runtime_scope(home, {}):
            assert runner._adapter_for_source(rebound) is primary
            assert runner._resolve_injection_adapter("telegram", rebound) is primary
            assert runner._is_user_authorized_for_source(rebound, allow_adapter_delegation=False) is primary_allowed
    finally:
        set_multiplex_active(previous_multiplex)
    del runner.adapters[Platform.TELEGRAM]
    assert runner._source_for_injection_target(target) is None


def test_process_watchers_and_checkpoint_preserve_the_same_original_target(route, tmp_path, monkeypatch):
    runner, adapter, entry, target, _, _ = route
    import tools.process_registry as registry_module
    monkeypatch.setattr(registry_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    runner._set_session_env(SessionContext(source=target.source, connected_platforms=[], home_channels={},
        session_key=entry.session_key, session_id=entry.session_id))
    registry = ProcessRegistry()
    process = registry._new_session(command="bounded-task", task_id="task", owner_task_id="task",
        session_key=entry.session_key, cwd=str(tmp_path), pid=4242,
        host_start_time=123.0, notify_on_complete=True)
    registry._running[process.id] = process
    _stamp_gateway_routing(process, get_session_env)
    _register_completion_watcher(registry, process, entry.session_key)
    assert registry.pending_watchers[0]["injection_target"] == target.to_dict()
    restored = ProcessRegistry()
    monkeypatch.setattr(restored, "_host_pid_is_ours", lambda *_: True)
    assert restored.recover_from_checkpoint() == 1
    watcher = restored.pending_watchers[0]
    assert watcher["injection_target"] == target.to_dict()
    entry.origin = replace(target.source, user_id="99", user_id_alt="alt-99")
    event = runner._build_process_completion_event(watcher, restored.get(process.id), process.id)
    assert runner._build_process_event_source(event) == target.source


@pytest.mark.asyncio
async def test_original_process_requester_rechecked_after_other_human_turn(route, monkeypatch):
    runner, adapter, entry, target, _, _ = route
    entry.origin = replace(target.source, user_id="99", user_id_alt="alt-99")
    event = {"type": "completion", "session_id": "owned-process", "session_key": target.session_key,
             "parent_session_id": target.session_id, "injection_target": target.to_dict(), "started_at": 1.0}
    assert await runner._deliver_completion_notification("result", event) is True
    assert adapter._pending_messages[target.session_key].source == target.source
    assert await runner._deliver_completion_notification("duplicate", event) is None
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "99")
    assert await runner._deliver_completion_notification("another", {**event, "session_id": "second-process"}) is None
    assert not runner._queued_events


@pytest.mark.asyncio
async def test_busy_adapter_keeps_independent_text_voice_document_burst_fifo(route):
    runner, adapter, entry, target, _, _ = route
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "interrupt"
    events = [MessageEvent(text=f"independent request {index}", source=target.source,
                          message_id=f"request-{index}", message_type=kind)
              for index, kind in enumerate((MessageType.TEXT, MessageType.TEXT,
                  MessageType.VOICE, MessageType.DOCUMENT, MessageType.VOICE))]
    started = asyncio.get_running_loop().time()
    for event in events:
        await adapter.handle_message(event)
    assert asyncio.get_running_loop().time() - started < adapter._busy_text_debounce_seconds
    queued = [adapter._pending_messages[entry.session_key], *runner._queued_events[entry.session_key]]
    assert queued == events
    assert len({id(event) for event in queued}) == 5
    assert not adapter._active_sessions[entry.session_key].is_set()
    assert not adapter._text_debounce
    adapter._message_handler.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_batching", [False, True])
async def test_completion_batching_reads_owner_profile_not_ambient(route, tmp_path, monkeypatch, owner_batching):
    from gateway.run import _profile_runtime_scope
    import hermes_cli.profiles as profiles

    runner, adapter, _, target, _, _ = route
    owner_home = tmp_path / "profiles" / "owner"
    ambient_home = tmp_path / "ambient"
    owner_home.mkdir(parents=True)
    ambient_home.mkdir()
    for home, batching in ((owner_home, owner_batching), (ambient_home, not owner_batching)):
        (home / "config.yaml").write_text(yaml.safe_dump({"display": {
            "background_process_completion_batching": batching}}))
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: tmp_path / "profiles")
    (owner_home / ".env").write_text("TELEGRAM_ALLOWED_USERS=42,alt-42\n")
    source = replace(target.source, profile="owner")
    runner.config.multiplex_profiles = True
    runner._profile_adapters = {"owner": {Platform.TELEGRAM: adapter}}
    owner_entry = runner.session_store.get_or_create_session(source)
    owner_target = InjectionTarget(owner_entry.session_key, owner_entry.session_id, source, "owner", "telegram")
    adapter._active_sessions[owner_target.session_key] = asyncio.Event()
    events = [{"type": "completion", "session_key": owner_target.session_key,
               "session_id": f"process-{index}", "started_at": float(index),
               "parent_session_id": owner_target.session_id,
               "injection_target": owner_target.to_dict()} for index in range(2)]
    with _profile_runtime_scope(ambient_home, {}):
        outcomes = await asyncio.gather(*(runner._enqueue_process_completion_notification(
            f"result-{index}", event) for index, event in enumerate(events)))
    assert outcomes == [True, True]
    delivered = [adapter._pending_messages[owner_target.session_key],
                 *runner._queued_events.get(owner_target.session_key, [])]
    assert len(delivered) == (1 if owner_batching else 2)
    if not owner_batching:
        assert sorted(event.text for event in delivered) == ["result-0", "result-1"]
    assert all(event.source == owner_target.source for event in delivered)
