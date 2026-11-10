# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Control-plane and fail-closed transport tests without loading NIXL or Neuron."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from verl_hardware_plugin.accelerators.neuron.weight_sync import nixl_transport
from verl_hardware_plugin.accelerators.neuron.weight_sync.nixl_transport import NixlEndpoint, NixlTopology, slot_address


def test_transport_refuses_cpu_and_cpu_slot():
    with pytest.raises(ValueError, match="real Trainium"):
        NixlEndpoint(None, "pair", "cpu", 0, sender=True)
    with pytest.raises(ValueError, match="Neuron allocation"):
        slot_address(torch.zeros(4))


def test_native_sync_is_scoped_to_the_assigned_device(monkeypatch):
    native = SimpleNamespace(current_device=lambda: 3, synchronize=Mock())
    monkeypatch.setattr(torch, "neuron", native, raising=False)
    nixl_transport.synchronize()
    native.synchronize.assert_called_once_with(3)


def test_topology_replica_bounds():
    topology = NixlTopology("localhost", 12345, 2, 2, 30)
    assert topology.receiver(1)["replica"] == 1
    with pytest.raises(ValueError):
        topology.receiver(2)


def endpoint(sender):
    return SimpleNamespace(
        sender=sender,
        lite=False,
        transfers={},
        remote="receiver",
        address=1024,
        remote_address=2048,
        registration=object(),
        buffer=torch.ones(4, dtype=torch.bfloat16),
        sequence=3,
        timeout_s=1,
        store=Mock(),
        agent=Mock(),
    )


@pytest.mark.parametrize("state", ["ERR", "PROC"])
def test_failed_write_never_signals_receiver(monkeypatch, state):
    monkeypatch.setattr(nixl_transport, "synchronize", Mock())
    instance = endpoint(True)
    instance.timeout_s = 0
    instance.agent.transfer.return_value = state
    with pytest.raises((RuntimeError, TimeoutError)):
        NixlEndpoint.push(instance, 3)
    instance.store.set.assert_not_called()
    assert instance.sequence == 3


def test_sender_requires_receiver_apply_acknowledgement(monkeypatch):
    monkeypatch.setattr(nixl_transport, "synchronize", Mock())
    instance = endpoint(True)
    instance.agent.transfer.return_value = "DONE"
    instance.store.get.return_value = b"apply failed"
    with pytest.raises(RuntimeError, match="receiver rejected"):
        NixlEndpoint.push(instance, 3)
    assert instance.sequence == 3
    instance.store.get.return_value = b"ok"
    NixlEndpoint.push(instance, 3)
    assert instance.sequence == 4
    instance.store.set.assert_called_with("written/3", b"ok")


def test_writes_only_payload_bytes_and_reuses_handles(monkeypatch):
    monkeypatch.setattr(nixl_transport, "synchronize", Mock())
    instance = endpoint(True)
    instance.agent.get_xfer_descs.side_effect = lambda descriptors, kind: descriptors
    handles = [object(), object(), object()]
    instance.agent.initialize_xfer.side_effect = handles
    instance.agent.transfer.return_value = "DONE"
    instance.store.get.return_value = b"ok"
    for numel in (4, 1, 3, 1, 4):
        NixlEndpoint.push(instance, numel)
    assert instance.agent.initialize_xfer.call_args_list == [
        call("WRITE", [(1024, size, 0)], [(2048, size, 0)], "receiver", b"") for size in (8, 2, 6)
    ]
    assert instance.agent.transfer.call_args_list == [call(handles[i]) for i in (0, 1, 2, 1, 0)]
    assert instance.sequence == 8
    instance.agent.register_memory.assert_not_called()
    instance.agent.deregister_memory.assert_not_called()


@pytest.mark.parametrize("numel", [-1, 0, 5])
def test_invalid_payload_never_starts_a_transfer(monkeypatch, numel):
    sync = Mock()
    monkeypatch.setattr(nixl_transport, "synchronize", sync)
    instance = endpoint(True)
    with pytest.raises(ValueError, match="payload"):
        NixlEndpoint.push(instance, numel)
    instance.agent.get_xfer_descs.assert_not_called()
    instance.agent.transfer.assert_not_called()
    instance.store.set.assert_not_called()
    sync.assert_not_called()
    assert instance.sequence == 3


def test_slot_registration_covers_full_allocation_without_starting_a_write(monkeypatch):
    monkeypatch.setattr(nixl_transport, "slot_address", lambda *args, **kwargs: (1024, 8))
    instance = endpoint(True)
    instance.buffer = None
    instance.device = "cpu"
    instance.store.get.side_effect = [b"[2048, 8]", b"remote metadata"]
    instance.agent.add_remote_agent.return_value = b"receiver"
    NixlEndpoint.register(instance, 4)
    buffer = instance.buffer
    NixlEndpoint.register(instance, 4)
    assert instance.buffer is buffer
    instance.agent.get_reg_descs.assert_called_once_with([(1024, 8, 0, "")], "VRAM")
    instance.agent.register_memory.assert_called_once()
    instance.agent.initialize_xfer.assert_not_called()
    assert (instance.address, instance.remote_address, instance.remote) == (1024, 2048, "receiver")
    with pytest.raises(ValueError, match="slot size cannot change"):
        NixlEndpoint.register(instance, 8)


def test_close_releases_every_cached_handle_before_memory():
    instance = endpoint(True)
    handles = [object(), object()]
    instance.transfers = dict(zip((2, 6), handles, strict=True))
    registration = instance.registration
    NixlEndpoint.close(instance)
    NixlEndpoint.close(instance)
    assert instance.agent.mock_calls == [
        call.release_xfer_handle(handles[0]),
        call.release_xfer_handle(handles[1]),
        call.remove_remote_agent("receiver"),
        call.deregister_memory(registration),
    ]
    assert instance.transfers == {}
    assert instance.buffer is instance.address is instance.remote_address is None


def test_receiver_acknowledges_only_after_apply_and_device_sync(monkeypatch):
    events = []
    monkeypatch.setattr(nixl_transport, "synchronize", lambda **kwargs: events.append("sync"))
    instance = endpoint(False)
    instance.store.set.side_effect = lambda *args: events.append("ack")
    assert NixlEndpoint.receive(instance, lambda slot: events.append("apply") or 4) == 4
    assert events == ["apply", "sync", "ack"]
    assert instance.sequence == 4


def test_receiver_failure_is_reported_without_advancing_sequence(monkeypatch):
    synchronize = Mock()
    monkeypatch.setattr(nixl_transport, "synchronize", synchronize)
    instance = endpoint(False)
    with pytest.raises(ValueError, match="sentinel"):
        NixlEndpoint.receive(instance, Mock(side_effect=ValueError("sentinel")))
    instance.store.set.assert_called_once_with("applied/3", b"error: sentinel")
    synchronize.assert_not_called()
    assert instance.sequence == 3
