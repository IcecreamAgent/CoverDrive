"""User-inactivity deadlines without suppliers, paid calls, or wall-clock waits."""

import asyncio
import time
from array import array

import pytest

from test_realtime import Socket, is_event, native as native, settings
from voice_agent.endpointing import EndpointDetector
from voice_agent.session import VoiceSession


class ClosingSocket(Socket):
    closed = None

    async def close(self, **values):
        self.closed = values


def expire(instance, seconds=61):
    instance._idle_last_activity = time.monotonic() - seconds
    instance._idle_activity.set()


async def start_cascade():
    socket = ClosingSocket()
    instance = VoiceSession(socket, settings(voice_provider="cascade"), speech_clients=False)

    async def enable():
        instance.duplex.enabled = True

    async def no_audio(audio):
        pass

    instance.duplex.enable = enable
    instance.handle_audio = no_audio
    runner = asyncio.create_task(instance.run())
    await socket.until(is_event("ready"))
    return socket, instance, runner


async def test_ready_ping_does_not_arm_but_listen_warns_times_out_and_closes():
    socket, instance, runner = await start_cascade()
    try:
        await socket.control({"type": "ping"})
        await socket.until(is_event("pong"))
        assert instance._idle_last_activity is None
        await socket.control({"type": "listen"})
        await socket.control({"type": "ping"})
        await socket.until(is_event("pong"))
        assert instance._idle_last_activity is not None
        expire(instance, 51)
        warning = (await socket.until(is_event("idle_warning")))[-1]
        assert warning["remaining_seconds"] <= 10
        assert "自动挂断" in warning["message"]
        expire(instance)
        await asyncio.wait_for(runner, 1)
        events = socket.drain()
        assert any(is_event("idle_timeout", reason="user_inactive")(event) for event in events)
        assert socket.closed == {"code": 1000, "reason": "user_inactive"}
        assert not instance.duplex.enabled
    finally:
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)


async def test_ping_silence_empty_pcm_repeated_listen_and_assistant_output_do_not_renew():
    socket, instance, runner = await start_cascade()
    try:
        instance.arm_idle_timeout()
        stamp = instance._idle_last_activity
        for audio in (b"", bytes(640), bytes(6400)):
            await socket.incoming.put({"type": "websocket.receive", "bytes": audio})
        await socket.control({"type": "listen"})
        await instance.send_global("assistant_delta", text="仍在回答。")
        await socket.control({"type": "ping"})
        await socket.until(is_event("pong"))
        assert instance._idle_last_activity == stamp
        expire(instance)
        await asyncio.wait_for(runner, 1)
    finally:
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)


async def test_valid_text_clears_warning_and_response_is_canceled_at_deadline():
    socket, instance, runner = await start_cascade()
    started, canceled = asyncio.Event(), asyncio.Event()

    async def respond(text):
        assert text == "继续体验"
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            canceled.set()

    instance.respond = respond
    try:
        instance.arm_idle_timeout()
        expire(instance, 51)
        await socket.until(is_event("idle_warning"))
        await socket.control({"type": "text", "text": "继续体验"})
        await asyncio.wait_for(started.wait(), 1)
        await socket.until(is_event("idle_resumed"))
        assert time.monotonic() - instance._idle_last_activity < 1
        expire(instance)
        await asyncio.wait_for(runner, 1)
        assert canceled.is_set()
        assert instance.response_task is None
    finally:
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)


@pytest.mark.parametrize("audio,speech,renews", [
    (bytes(6400), True, False),
    (array("h", [3000, -3000] * 1600).tobytes(), False, False),
    (array("h", [3000, -3000] * 1600).tobytes(), True, True),
    (b"x", True, False),
    (bytes(64002), True, False),
])
def test_only_confirmed_local_speech_renews_not_energy_noise_or_invalid_frames(audio, speech, renews):
    instance = VoiceSession(ClosingSocket(), settings(), speech_clients=False)
    instance._idle_detector = EndpointDetector(instance.settings, classifier=lambda *_: speech)
    instance.arm_idle_timeout()
    stamp = instance._idle_last_activity
    instance.observe_user_audio(audio)
    assert (instance._idle_last_activity > stamp) is renews


async def test_native_final_text_renews_while_partial_transcript_does_not(native):
    _, instance, _, _ = native
    instance.arm_idle_timeout()
    stamp = instance._idle_last_activity
    await instance.handle_native_event({
        "type": "conversation.item.input_audio_transcription.started", "item_id": "idle-q",
    })
    await instance.handle_native_event({
        "type": "conversation.item.input_audio_transcription.delta", "item_id": "idle-q", "delta": "继续",
    })
    assert instance._idle_last_activity == stamp
    await instance.handle_native_event({
        "type": "conversation.item.input_audio_transcription.completed", "item_id": "idle-q", "text": "继续体验",
    })
    assert instance._idle_last_activity > stamp


async def test_native_timeout_cancels_blocked_handshake_even_without_provider_events(native):
    socket, instance, _, _ = native
    entered, canceled = asyncio.Event(), asyncio.Event()
    closed = []

    async def close(**values):
        closed.append(values)

    async def stalled():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            canceled.set()

    socket.close = close
    instance.enable_native = stalled
    runner = asyncio.create_task(instance.run())
    try:
        await socket.control({"type": "listen"})
        await asyncio.wait_for(entered.wait(), 1)
        expire(instance)
        await asyncio.wait_for(runner, 1)
        assert canceled.is_set()
        assert instance.native is None and instance.background_closed
        assert closed == [{"code": 1000, "reason": "user_inactive"}]
    finally:
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)


async def test_native_timeout_closes_supplier_and_cancels_owned_work(native):
    socket, instance, _, state = native
    closed = []

    async def close(**values):
        closed.append(values)

    async def work():
        await asyncio.Event().wait()

    socket.close = close
    owned = [asyncio.create_task(work()) for _ in range(3)]
    instance.response_task, instance.tool_task, instance.background_loop = owned
    instance.arm_idle_timeout()
    runner = asyncio.create_task(instance.run())
    expire(instance)
    await asyncio.wait_for(runner, 1)
    assert all(task.done() and task.cancelled() for task in owned)
    assert state.closed == 1
    assert instance.native is instance.native_context is instance.native_task is None
    assert instance.background_closed and closed


async def test_receive_failure_propagates_and_does_not_leave_idle_monitor_running():
    socket = ClosingSocket()
    instance = VoiceSession(socket, settings(), speech_clients=False)

    async def failure():
        raise RuntimeError("isolated receive failure")

    socket.receive = failure
    instance.arm_idle_timeout()
    with pytest.raises(RuntimeError, match="isolated receive failure"):
        await instance.run()
    assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()
                and not task.done() and "_watch_idle" in str(task.get_coro())]


@pytest.mark.parametrize("disconnect", [True, False])
async def test_disconnect_or_external_cancellation_releases_monitor_and_owned_tasks(disconnect):
    socket, instance, runner = await start_cascade()
    instance.arm_idle_timeout()
    if disconnect:
        await socket.incoming.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(runner, 1)
    else:
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner
    assert not instance.duplex.enabled
    assert socket.closed is None
    assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()
                and not task.done() and "_watch_idle" in str(task.get_coro())]
