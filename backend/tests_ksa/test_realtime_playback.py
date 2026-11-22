"""Browser playback ownership across native turn changes, without network I/O."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from test_realtime import is_event, utterance
from test_realtime import native as native


@asynccontextmanager
async def controls(native):
    socket, instance, _, _ = native
    runner = asyncio.create_task(instance.run())
    await socket.until(is_event("ready"))
    try:
        yield socket, instance
    finally:
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)


async def playback(socket, turn_id, playing):
    await socket.control({"type": "playback", "turn_id": turn_id, "playing": playing})
    # A pong establishes that run() has handled the preceding control message.
    await socket.control({"type": "ping"})
    await socket.until(is_event("pong"))


async def test_old_playback_can_drain_after_new_turn_with_no_audio(native):
    async with controls(native) as (socket, instance):
        previous = await utterance(instance)
        previous.finished = True
        await playback(socket, previous.turn_id, True)
        current = await utterance(instance, item="q2", text="先等等")
        current.finished = True
        assert current is not previous
        assert instance.duplex.playback_active
        assert not instance._notification_safe()

        await playback(socket, previous.turn_id, False)

        assert not instance.duplex.playback_active
        assert instance._notification_safe()


async def test_old_playback_end_cannot_clear_new_turn_audio(native):
    async with controls(native) as (socket, instance):
        previous = await utterance(instance)
        previous.finished = True
        await playback(socket, previous.turn_id, True)
        current = await utterance(instance, item="q2", text="继续说吧")
        current.finished = True
        await playback(socket, current.turn_id, True)

        await playback(socket, previous.turn_id, False)

        assert instance.duplex.playback_active
        assert not instance._notification_safe()
        await playback(socket, current.turn_id, False)
        assert not instance.duplex.playback_active
        assert instance._notification_safe()


async def test_late_old_playback_start_cannot_take_new_turn_ownership(native):
    async with controls(native) as (socket, instance):
        previous = await utterance(instance)
        current = await utterance(instance, item="q2", text="继续说吧")
        await playback(socket, current.turn_id, True)

        await playback(socket, previous.turn_id, True)
        await playback(socket, previous.turn_id, False)

        assert instance.duplex.playback_active
        await playback(socket, current.turn_id, False)
        assert not instance.duplex.playback_active


@pytest.mark.parametrize("playing", [None, 0, 1, "false", "true"])
async def test_malformed_playback_flag_cannot_clear_owned_audio(native, playing):
    async with controls(native) as (socket, instance):
        current = await utterance(instance)
        await playback(socket, current.turn_id, True)

        await playback(socket, current.turn_id, playing)

        assert instance.duplex.playback_active
        await playback(socket, current.turn_id, False)
        assert not instance.duplex.playback_active
