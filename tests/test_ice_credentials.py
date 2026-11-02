"""The supported ICE dependency must defer checks until credentials exist."""

import asyncio

import pytest

pytest.importorskip("aioice")
import aioice.ice


def test_remote_check_before_transport_start_preserves_handshake(monkeypatch):
    # Advertise only loopback while still exercising authenticated STUN and UDP.
    monkeypatch.setattr(aioice.ice, "get_host_addresses", lambda **_: ["127.0.0.1"])

    async def scenario():
        controlling = aioice.ice.Connection(ice_controlling=True)
        controlled = aioice.ice.Connection(ice_controlling=False)
        incoming = asyncio.Event()
        original_receive = controlling.request_received
        tasks = []

        def receive(*args):
            original_receive(*args)
            incoming.set()

        controlling.request_received = receive
        try:
            await asyncio.gather(
                controlling.gather_candidates(), controlled.gather_candidates()
            )
            for candidate in controlled.local_candidates:
                await controlling.add_remote_candidate(candidate)
            for candidate in controlling.local_candidates:
                await controlled.add_remote_candidate(candidate)
            await controlling.add_remote_candidate(None)
            await controlled.add_remote_candidate(None)
            controlled.remote_username = controlling.local_username
            controlled.remote_password = controlling.local_password
            tasks.append(asyncio.create_task(controlled.connect()))

            # aiortc sets remote credentials in RTCIceTransport.start(), after
            # receiving SDP candidates. Force the peer's first check into that
            # interval instead of depending on OS scheduling to expose it.
            await asyncio.wait_for(incoming.wait(), 5)
            await asyncio.sleep(0)
            errors = [
                pair.task.exception()
                for pair in controlling._check_list
                if pair.task and pair.task.done() and not pair.task.cancelled()
            ]
            assert not any(errors), errors
            controlling.remote_username = controlled.local_username
            controlling.remote_password = controlled.local_password
            tasks.append(asyncio.create_task(controlling.connect()))
            await asyncio.wait_for(asyncio.gather(*tasks), 5)
            await controlled.send(b"synthetic-media")
            assert (
                await asyncio.wait_for(controlling.recv(), 5)
                == b"synthetic-media"
            )
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(controlling.close(), controlled.close())

    asyncio.run(scenario())
