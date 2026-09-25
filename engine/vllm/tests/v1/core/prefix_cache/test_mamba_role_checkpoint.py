# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" check-point at a prompt's last turn-opening token (engine vllm 101).

Agent prompts end with a turn the next request replaces, so the follow-up
shares the prompt only up to that turn's opening token. Without a state there
the follow-up falls back to an older, lower state. Every test drives the real
``Scheduler._mamba_block_aligned_split`` / ``_role_checkpoint_positions`` and the
real ``KVCacheManager``; no boundary is hard-coded except as an expectation.
"""

from types import SimpleNamespace

import pytest

from tests.v1.core.test_prefix_caching import (
    _make_hybrid_kv_cache_config,
    make_kv_cache_manager,
    make_request,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler

ROLE = 1_000_001  # stands in for GLM's <|user|>
OTHER_ROLE = 1_000_002  # <|observation|>
ASSISTANT = 1_000_003
PREFIX = list(range(1, 20_001))


def _manager(
    block_size,
    hash_block_size,
    role_ids=(ROLE, OTHER_ROLE),
    mode="mamba_align",
    enable_caching=True,
):
    init_none_hash(sha256)
    config = _make_hybrid_kv_cache_config(block_size, 8192, ["full", mode])
    return make_kv_cache_manager(
        kv_cache_config=config,
        max_model_len=1 << 20,
        enable_caching=enable_caching,
        hash_block_size=hash_block_size,
        use_eagle=True,
        retention_interval=0,  # the `vllm serve` default: sparse retention
        mamba_role_checkpoint_token_ids=role_ids,
    )


def _stub(manager, block_size, hash_block_size, block_drop=True):
    """``self`` for the real scheduler methods, gated as ``Scheduler.__init__``."""
    partial_hit = (
        hash_block_size < block_size and manager.coordinator.enable_partial_hash_hits
    )
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        max_num_scheduled_tokens=1 << 20,
        use_eagle=True,
        use_eagle_block_drop=block_drop,
        hash_block_size=hash_block_size,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_partial_cache_hit=partial_hit,
        mamba_shared_prefix_checkpoint=False,
        mamba_role_checkpoint=manager.mamba_role_checkpoint,
        kv_cache_manager=manager,
    )


def _prefill(manager, stub, request) -> list[int]:
    """Admit and schedule ``request`` to completion; return the chunk ends."""
    if stub.mamba_role_checkpoint:
        request.role_boundary, request.role_checkpoint = (
            Scheduler._role_checkpoint_positions(stub, request)
        )
    blocks, local, junction = manager.get_computed_blocks(request)
    request.shared_prefix_boundary = junction
    ends, first = [], True
    while request.num_computed_tokens < request.num_tokens:
        new_local = local if first else 0
        start = request.num_computed_tokens + new_local
        if start >= request.num_tokens:
            break
        num_new = Scheduler._mamba_block_aligned_split(
            stub, request, request.num_tokens - start, new_local, 0
        )
        assert num_new > 0
        assert (
            manager.allocate_slots(
                request,
                num_new,
                num_new_computed_tokens=new_local,
                new_computed_blocks=blocks if first else None,
                has_scheduled_reqs=False,
            )
            is not None
        )
        request.num_computed_tokens = start + num_new
        ends.append(request.num_computed_tokens)
        _, retained = manager.take_kv_cache_block_copies()
        if retained:
            manager.block_pool.free_blocks(retained)
        manager.new_step_starts()
        first = False
    return ends


def _decode(manager, request, num_tokens):
    """Generate ``num_tokens`` one step at a time, as the scheduler does."""
    request.max_tokens = request.sampling_params.max_tokens = num_tokens
    for _ in range(num_tokens):
        request.append_output_token_ids(7)
        assert manager.allocate_slots(request, 1, has_scheduled_reqs=False) is not None
        request.num_computed_tokens += 1
        _, retained = manager.take_kv_cache_block_copies()
        if retained:
            manager.block_pool.free_blocks(retained)
        manager.new_step_starts()


def _turn(shared, suffix_len, opener=ROLE):
    """``shared`` tokens, then a turn opened by ``opener`` the next prompt drops."""
    return PREFIX[:shared] + [opener] + [-7] * suffix_len


def _follow_up_hit(manager, shared, hash_block_size):
    follower = make_request(
        "follower", PREFIX[:shared] + [ASSISTANT] + [-9] * 600, hash_block_size, sha256
    )
    return manager.get_computed_blocks(follower)[1]


@pytest.mark.parametrize(
    "shared,suffix",
    [
        (2021, 430),  # check-point inside a block
        (2400, 430),
        (2560, 430),  # opener on the block grid, check-point one unit below it
        (2600, 430),  # check-point exactly on the block grid (2560)
        (2021, 538),  # prompt tail on the block grid (2560): no tail partial key
    ],
)
def test_follow_up_resumes_at_the_role_checkpoint(shared, suffix):
    """The owner's chunk stops at the predicted junction and caches it there."""
    block_size, hash_block_size = 512, 32
    manager = _manager(block_size, hash_block_size)
    stub = _stub(manager, block_size, hash_block_size)
    assert stub.mamba_partial_cache_hit and stub.mamba_role_checkpoint

    owner = make_request("owner", _turn(shared, suffix), hash_block_size, sha256)
    ends = _prefill(manager, stub, owner)
    expected = (shared // hash_block_size - 1) * hash_block_size  # EAGLE drops a unit
    assert owner.role_checkpoint == expected
    assert expected in ends

    assert _follow_up_hit(manager, shared, hash_block_size) == expected


@pytest.mark.parametrize(
    "shared,suffix,hash_block_size,num_output",
    [
        (2200, 100, 32, 300),
        (2120, 430, 64, 128),  # agent-sized reminder, check-point on the grid
    ],
)
def test_role_checkpoint_outlives_decode_and_owner(
    shared, suffix, hash_block_size, num_output
):
    """Decode fills the role block; promoting it to a full block drops all its
    keys, so the role key must come back, and must outlive the owner."""
    block_size = 512
    manager = _manager(block_size, hash_block_size)
    stub = _stub(manager, block_size, hash_block_size)
    owner = make_request("owner", _turn(shared, suffix), hash_block_size, sha256)
    _prefill(manager, stub, owner)
    expected = owner.role_checkpoint
    role_block_end = (owner.role_boundary // block_size + 1) * block_size
    assert owner.num_tokens < role_block_end <= owner.num_tokens + num_output
    assert _follow_up_hit(manager, shared, hash_block_size) == expected

    _decode(manager, owner, num_output)
    assert _follow_up_hit(manager, shared, hash_block_size) == expected
    manager.free(owner)
    assert _follow_up_hit(manager, shared, hash_block_size) == expected


@pytest.mark.parametrize("shared", [2021, 2400, 2560])
def test_disabled_keeps_base_behaviour(shared):
    """Without token ids nothing changes: same chunk ends, lower follow-up hit."""
    block_size, hash_block_size = 512, 32
    base = _manager(block_size, hash_block_size, role_ids=())
    base_stub = _stub(base, block_size, hash_block_size)
    assert not base_stub.mamba_role_checkpoint
    owner = make_request("owner", _turn(shared, 430), hash_block_size, sha256)
    base_ends = _prefill(base, base_stub, owner)
    assert owner.role_checkpoint == 0
    base_hit = _follow_up_hit(base, shared, hash_block_size)

    on = _manager(block_size, hash_block_size)
    on_stub = _stub(on, block_size, hash_block_size)
    owner_on = make_request("owner", _turn(shared, 430), hash_block_size, sha256)
    on_ends = _prefill(on, on_stub, owner_on)
    assert set(base_ends) <= set(on_ends), "the check-point only adds a stop"
    assert _follow_up_hit(on, shared, hash_block_size) > base_hit


def test_last_opener_of_either_kind_is_used():
    block_size, hash_block_size = 512, 32
    manager = _manager(block_size, hash_block_size)
    stub = _stub(manager, block_size, hash_block_size)
    prompt = PREFIX[:1000] + [ROLE] + PREFIX[1000:1800] + [OTHER_ROLE] + [-7] * 300
    request = make_request("r", prompt, hash_block_size, sha256)
    last_opener = prompt.index(OTHER_ROLE)
    assert (
        Scheduler._role_checkpoint_positions(stub, request)[1]
        == (last_opener // hash_block_size - 1) * hash_block_size
    )


def test_without_eagle_block_drop_the_unit_is_not_dropped():
    block_size, hash_block_size = 512, 32
    manager = _manager(block_size, hash_block_size)
    stub = _stub(manager, block_size, hash_block_size, block_drop=False)
    request = make_request("r", _turn(2021, 430), hash_block_size, sha256)
    assert Scheduler._role_checkpoint_positions(stub, request)[1] == 2016


@pytest.mark.parametrize(
    "prompt",
    [
        PREFIX[:3000],  # no turn opener at all
        PREFIX[:2990] + [ROLE] + [-7] * 5,  # opener inside the prompt-tail unit
    ],
)
def test_no_checkpoint_when_nothing_to_add(prompt):
    block_size, hash_block_size = 512, 32
    manager = _manager(block_size, hash_block_size)
    stub = _stub(manager, block_size, hash_block_size)
    request = make_request("r", prompt, hash_block_size, sha256)
    assert Scheduler._role_checkpoint_positions(stub, request)[1] == 0


@pytest.mark.parametrize(
    "block_size,hash_block_size,mode,enable_caching",
    [
        (512, 512, "mamba_align", True),  # no prefix-match unit finer than the block
        (512, 32, "full", True),  # no Mamba group
        (512, 32, "mamba_align", False),  # no prefix caching
    ],
)
def test_unsupported_configurations_refuse_to_start(
    block_size, hash_block_size, mode, enable_caching
):
    with pytest.raises(ValueError, match="VLLM_AX_MAMBA_ROLE_CHECKPOINT_TOKEN_IDS"):
        _manager(block_size, hash_block_size, mode=mode, enable_caching=enable_caching)
