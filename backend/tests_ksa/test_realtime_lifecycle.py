"""Native connection ownership under overlapping browser actions, without network I/O."""

import asyncio
from contextlib import asynccontextmanager
from types import MethodType, SimpleNamespace

import pytest
from test_realtime import Socket, is_event, settings
from test_realtime import native as native

import voice_agent.realtime as realtime
from voice_agent.providers.seeduplex import SeeduplexStream


@pytest.fixture
async def lifecycle(monkeypatch):
    state = SimpleNamespace(
        opened=0, closed=0, streams=[], receivers=[], history_error=None, mute_error=None,
        entered=asyncio.Event(), enter_release=asyncio.Event(),
        history_entered=asyncio.Event(), history_release=asyncio.Event(),
        closing=asyncio.Event(), close_release=asyncio.Event(),
    )
    state.enter_release.set()
    state.history_release.set()
    state.close_release.set()

    class Stream:
        def __init__(self):
            self.events = asyncio.Queue()
            self.reading = False
            self.started = asyncio.Event()
            self.muted = []

        async def receive(self):
            self.reading = True
            self.started.set()
            state.receivers.append(asyncio.current_task())
            try:
                while True:
                    event = await self.events.get()
                    if isinstance(event, BaseException):
                        raise event
                    yield event
            finally:
                self.reading = False

        async def add_history(self, history):
            state.history_entered.set()
            await state.history_release.wait()
            if state.history_error:
                raise state.history_error

        async def set_muted(self, muted):
            if state.mute_error:
                raise state.mute_error
            self.muted.append(muted)

    class Client:
        @asynccontextmanager
        async def stream(self, **kwargs):
            stream = Stream()
            state.streams.append(stream)
            state.opened += 1
            state.entered.set()
            try:
                await state.enter_release.wait()
                # Historical Q&A is now part of session.create initialization,
                # before a connection can be published to the audio receiver.
                if kwargs.get("history"):
                    await stream.add_history(kwargs["history"])
                yield stream
            finally:
                # A real provider needs its reader to stop before close can
                # consume session.closed. A receiver closing itself must do so too.
                assert not stream.reading
                state.closing.set()
                await state.close_release.wait()
                state.closed += 1

    monkeypatch.setattr(realtime, "SeeduplexClient", lambda _: Client())
    monkeypatch.setattr(realtime, "DeliveryPlanner", lambda _: object())
    socket = Socket()
    instance = realtime.RealtimeVoiceSession(socket, settings(background_tasks_enabled=False))
    try:
        yield socket, instance, state
    finally:
        state.enter_release.set()
        state.history_release.set()
        state.close_release.set()
        await asyncio.wait_for(instance.close_native(), 1)
        assert all(task.done() for task in state.receivers)


async def test_simultaneous_first_connect_and_listen_own_one_context_and_reader(lifecycle):
    _, instance, state = lifecycle
    state.enter_release.clear()
    typed = asyncio.create_task(instance._connect_native())
    await state.entered.wait()
    listen = asyncio.create_task(instance.enable_native())
    await asyncio.sleep(0)
    assert state.opened == 1 and instance.native is None
    state.enter_release.set()
    await asyncio.wait_for(asyncio.gather(typed, listen), 1)
    await state.streams[0].started.wait()
    assert state.opened == 1 and len(state.receivers) == 1
    assert instance.listening
    await instance.disable_native()
    await instance.enable_native()
    assert state.opened == 1 and state.streams[0].muted == [False, True, False]
    await instance.close_native()
    assert state.closed == 1 and state.receivers[0].done()


@pytest.mark.parametrize("cancelled", [False, True])
async def test_failed_or_cancelled_history_initialization_closes_unpublished_context(
    lifecycle, cancelled,
):
    _, instance, state = lifecycle
    instance.public_history = [("问题", "答案")]
    state.history_release.clear()
    pending = asyncio.create_task(instance._connect_native())
    await state.history_entered.wait()
    assert instance.native is None and instance.native_context is None
    if cancelled:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        state.history_error = RuntimeError("isolated history failure")
        state.history_release.set()
        with pytest.raises(RuntimeError, match="isolated history failure"):
            await pending
    assert state.opened == state.closed == 1
    assert instance.native_task is None and not state.receivers
    # Only a new explicit request reconnects; failed initialization does not retry.
    state.history_error = None
    state.history_release.set()
    await instance._connect_native()
    assert state.opened == 2 and instance.native is state.streams[1]


async def test_cancelled_handshake_is_not_published_or_retried(lifecycle):
    _, instance, state = lifecycle
    state.enter_release.clear()
    pending = asyncio.create_task(instance._connect_native())
    await state.entered.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert state.opened == state.closed == 1
    assert instance.native is instance.native_context is instance.native_task is None


async def test_close_during_handshake_waits_then_closes_the_initialized_connection(lifecycle):
    _, instance, state = lifecycle
    state.enter_release.clear()
    opening = asyncio.create_task(instance._connect_native())
    await state.entered.wait()
    closing = asyncio.create_task(instance.close_native())
    await asyncio.sleep(0)
    assert not closing.done()
    state.enter_release.set()
    await asyncio.wait_for(asyncio.gather(opening, closing), 1)
    assert state.opened == state.closed == 1
    assert instance.native is instance.native_context is instance.native_task is None


async def test_new_connect_waits_until_old_context_finishes_closing(lifecycle):
    _, instance, state = lifecycle
    await instance._connect_native()
    state.close_release.clear()
    closing = asyncio.create_task(instance.close_native())
    await state.closing.wait()
    opening = asyncio.create_task(instance._connect_native())
    await asyncio.sleep(0)
    assert state.opened == 1 and not opening.done()
    state.close_release.set()
    await asyncio.wait_for(asyncio.gather(closing, opening), 1)
    assert state.opened == 2 and state.closed == 1
    assert instance.native is state.streams[1]


async def test_failed_listen_cleanup_cannot_close_a_waiting_new_connection(lifecycle):
    socket, instance, state = lifecycle
    state.mute_error = RuntimeError("isolated mute failure")
    state.close_release.clear()
    listening = asyncio.create_task(instance.enable_native())
    await state.closing.wait()
    opening = asyncio.create_task(instance._connect_native())
    await asyncio.sleep(0)
    assert state.opened == 1 and not opening.done()
    state.mute_error = None
    state.close_release.set()
    await asyncio.wait_for(asyncio.gather(listening, opening), 1)
    assert state.opened == 2 and state.closed == 1
    assert instance.native is state.streams[1]
    assert any(is_event("error", stage="seeduplex")(e) for e in socket.drain())


async def test_cancelled_connect_waiter_does_not_cancel_the_handshake_owner(lifecycle):
    _, instance, state = lifecycle
    state.enter_release.clear()
    owner = asyncio.create_task(instance._connect_native())
    await state.entered.wait()
    waiter = asyncio.create_task(instance._connect_native())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not owner.done() and state.closed == 0
    state.enter_release.set()
    await asyncio.wait_for(owner, 1)
    assert state.opened == 1 and instance.native is state.streams[0]


async def test_receiver_failure_closes_its_context_without_cancelling_or_awaiting_itself(lifecycle):
    socket, instance, state = lifecycle
    await instance.enable_native()
    receiver = instance.native_task
    stream = state.streams[0]
    await stream.started.wait()
    await stream.events.put(RuntimeError("isolated provider failure"))
    await asyncio.wait_for(receiver, 1)
    assert not receiver.cancelled() and receiver.exception() is None
    assert state.closed == 1 and not stream.reading
    assert instance.native is instance.native_context is instance.native_task is None
    assert not instance.listening
    assert any(is_event("error", stage="seeduplex")(e) for e in socket.drain())


async def test_cancelled_teardown_still_exits_the_detached_context(lifecycle, monkeypatch):
    _, instance, state = lifecycle
    await instance._connect_native()
    cleaning = asyncio.Event()
    original = instance.background.close

    async def cleanup():
        await original()
        cleaning.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(instance.background, "close", cleanup)
    closing = asyncio.create_task(instance.close_native())
    await cleaning.wait()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    monkeypatch.setattr(instance.background, "close", original)
    assert state.closed == 1
    assert instance.native is instance.native_context is instance.native_task is None


async def test_old_receiver_failure_does_not_retire_replacement_connection(lifecycle):
    socket, instance, state = lifecycle
    await instance._connect_native()
    old = instance.native
    await instance.close_native()
    await instance.enable_native()
    socket.drain()
    await old.events.put(RuntimeError("obsolete provider failure"))
    await instance.receive_native(old)
    assert instance.native is state.streams[1] and instance.listening
    assert state.closed == 1 and socket.drain() == []


async def test_listen_waits_for_typed_request_mute_before_unmuting_the_provider(lifecycle):
    _, instance, state = lifecycle
    await instance._connect_native()
    instance.settings.background_tasks_enabled = True
    stream = state.streams[0]
    stream._muted = False
    muting, release, delegating = asyncio.Event(), asyncio.Event(), asyncio.Event()
    wire = []

    async def send(kind, **kwargs):
        if kind == "input_audio_mute.commit":
            muting.set()
            await release.wait()
        wire.append(kind)

    # Use the real provider's cached-state optimization, with only wire I/O
    # replaced. An overlapping unmute must not observe its stale False value.
    stream._send = send
    stream.set_muted = MethodType(SeeduplexStream.set_muted, stream)

    async def delegate(**kwargs):
        delegating.set()
        await asyncio.Event().wait()

    instance.background.start_delegation = delegate
    response = asyncio.create_task(instance.respond("三加四"))
    listening = None
    try:
        await asyncio.wait_for(muting.wait(), 1)
        listening = asyncio.create_task(instance.enable_native())
        await asyncio.sleep(0)
        assert not listening.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(listening, delegating.wait()), 1)
        assert instance.listening and not stream._muted
        assert wire == ["input_audio_mute.commit", "input_audio_unmute.commit"]
    finally:
        release.set()
        response.cancel()
        if listening is not None:
            listening.cancel()
        await asyncio.gather(
            *[task for task in (response, listening) if task is not None],
            return_exceptions=True,
        )


@pytest.mark.parametrize("native", [True], indirect=True)
async def test_browser_text_and_listen_during_first_handshake_complete_on_one_connection(
    native, monkeypatch,
):
    socket, previous, _, state = native
    await previous.close_native()
    socket.drain()
    instance = type(previous)(socket, previous.settings)
    state.uses_tools = False
    state.agent_results["三加四"] = "七。"
    entered, release = asyncio.Event(), asyncio.Event()
    original = instance.native_client.stream

    @asynccontextmanager
    async def delayed(**kwargs):
        entered.set()
        await release.wait()
        async with original(**kwargs) as stream:
            yield stream

    monkeypatch.setattr(instance.native_client, "stream", delayed)
    before = state.opened
    session = asyncio.create_task(instance.run())
    try:
        await socket.until(is_event("ready"))
        await socket.control({"type": "text", "text": "三加四", "turn_id": "typed"})
        await entered.wait()
        await socket.control({"type": "listen"})
        await asyncio.sleep(0)
        release.set()
        observed = []
        async with asyncio.timeout(2):
            while not (
                any(is_event("listening", enabled=True)(e) for e in observed)
                and any(is_event("assistant_done", text="七。")(e) for e in observed)
                and any(is_event("audio_end")(e) for e in observed)
            ):
                observed.append(await socket.outgoing.get())
        assert state.opened == before + 1
        assert any(isinstance(event, bytes) and event for event in observed)
        assert not any(is_event("error")(e) for e in observed)
    finally:
        release.set()
        await socket.incoming.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(session, 1)
    assert state.closed == state.opened
