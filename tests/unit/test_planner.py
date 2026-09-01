# SPDX-License-Identifier: Apache-2.0
"""Tests for the static fit planner (per-file declining chunk budgets)."""

import random

import pytest

from fastsafetensors import SafeTensorsMetadata
from fastsafetensors import cpp as fstcpp
from fastsafetensors._planner import (
    BudgetInfeasibleError,
    FileWeightStats,
    collect_file_stats,
    pipeline_depth,
    plan_file_budgets,
)

GiB = 1024**3
MiB = 1024**2


def _st(path, kept, span=None, largest=None):
    return FileWeightStats(
        path,
        kept,
        span if span is not None else kept,
        largest if largest is not None else kept // 4 or kept,
    )


# ---- pipeline depth ----


def test_pipeline_depth_mapping():
    assert pipeline_depth(-1) == 1
    assert pipeline_depth(0) == 2
    assert pipeline_depth(1) == 3
    assert pipeline_depth(4) == 6


# ---- copier-supplied transient multiplier ----


def test_chunk_transient_multiplier_default_refuses():
    # Like set_chunk, the interface default refuses rather than guessing, so a
    # copier that cannot chunk can never silently under-count the plan.
    from fastsafetensors.copier import CopierInterface, GdsFileCopier

    for cls in (CopierInterface, GdsFileCopier):
        with pytest.raises(NotImplementedError, match="chunk_transient_multiplier"):
            cls.chunk_transient_multiplier(["f0"])


def test_chunking_copiers_declare_their_transient_cost():
    # The two overrides have to travel together: a copier that implements
    # set_chunk but inherits the refusing default would be rejected by
    # device_memory_budget despite being able to chunk, and one that declares
    # a cost without implementing set_chunk would be planned for and then fail.
    from fastsafetensors.copier import CopierInterface, get_copier_class
    from fastsafetensors.copier.registry import _copier_class_registry

    # Iterate the registry rather than a literal list: dropping the class
    # argument from a @register_copier_constructor would make a hardcoded
    # lookup return CopierInterface and satisfy the invariant vacuously, while
    # breaking device_memory_budget for every user of that copier.
    assert {"gds", "nogds", "unified", "3fs"} <= set(_copier_class_registry)
    for name, registered in sorted(_copier_class_registry.items()):
        cls = get_copier_class(name)
        assert cls is registered is not CopierInterface, name
        chunks = cls.set_chunk is not CopierInterface.set_chunk
        # chunk_transient_multiplier is a classmethod: attribute access builds
        # a fresh bound method every time, so compare the underlying functions.
        declares = (
            cls.chunk_transient_multiplier.__func__
            is not CopierInterface.chunk_transient_multiplier.__func__
        )
        assert chunks == declares, (
            f"{name}: set_chunk override={chunks} but "
            f"chunk_transient_multiplier override={declares}"
        )
        if declares:
            assert cls.chunk_transient_multiplier(["f0", "f1"]) >= 1
    # Exact, not just >= 1: silently bumping nogds to 2 would halve every
    # budget on the default CPU path without failing anything else.
    assert get_copier_class("nogds").chunk_transient_multiplier(["f0"]) == 1


def test_copier_class_follows_factory_fallback():
    """A factory may delegate to another copier's factory -- gds hands off to
    nogds/unified on a host without cuFile (any GPU box without GDS, i.e. the
    common case on the default nogds=False path). The class the planner reads
    must be the delegate's, not the requested type's: resolving it from the
    type name instead makes device_memory_budget die with 'GdsFileCopier does
    not implement sub-file chunking' while max_batch_bytes, which never
    consults the class, keeps working on the very same loader."""
    from fastsafetensors.copier import CopierInterface, copier_class_of
    from fastsafetensors.copier.registry import (
        _copier_class_registry,
        _copier_registry,
        create_copier_constructor,
        register_copier_constructor,
    )

    class _Delegate(CopierInterface):
        @classmethod
        def chunk_transient_multiplier(cls, paths):
            return 1

    class _Front(CopierInterface):
        pass

    saved = dict(_copier_registry), dict(_copier_class_registry)
    try:

        @register_copier_constructor("_test_delegate", _Delegate)
        def _delegate_factory(device, **kwargs):
            def construct(metadata, device, framework):
                raise AssertionError("not constructed in this test")

            return construct

        @register_copier_constructor("_test_front", _Front)
        def _front_factory(device, **kwargs):
            return _delegate_factory(device, **kwargs)  # hand off, like gds

        assert copier_class_of(create_copier_constructor("_test_delegate", None)) is (
            _Delegate
        )
        # The delegate tags first, so it wins over the front's own class.
        front = create_copier_constructor("_test_front", None)
        assert copier_class_of(front) is _Delegate
        assert copier_class_of(front).chunk_transient_multiplier(["f"]) == 1
    finally:
        _copier_registry.clear()
        _copier_registry.update(saved[0])
        _copier_class_registry.clear()
        _copier_class_registry.update(saved[1])


def test_chunk_transient_multiplier_unified_tracks_reader_path(monkeypatch):
    from fastsafetensors.copier import UnifiedMemCopier

    if getattr(fstcpp, "dma_load_runs", None) is None:
        pytest.skip("built without the O_DIRECT reader; only the fallback exists")

    # O_DIRECT reader available and usable -> chunk staged once.
    monkeypatch.setenv("FASTSAFETENSORS_ODIRECT", "1")
    monkeypatch.setenv("FASTSAFETENSORS_DMA_THREADS", "8")
    assert UnifiedMemCopier.chunk_transient_multiplier(["f0"]) == 1

    # Reader disabled -> mmap+pin fallback pins the chunk alongside the buffer.
    monkeypatch.setenv("FASTSAFETENSORS_DMA_THREADS", "0")
    assert UnifiedMemCopier.chunk_transient_multiplier(["f0"]) == 2

    # Network filesystem -> reader skipped for the same reason.
    monkeypatch.setenv("FASTSAFETENSORS_DMA_THREADS", "8")
    monkeypatch.setenv("FASTSAFETENSORS_ODIRECT", "0")
    assert UnifiedMemCopier.chunk_transient_multiplier(["f0"]) == 2


# ---- planner math ----


def test_budgets_decline_monotonically():
    stats = [_st(f"f{i}", 4 * GiB) for i in range(8)]
    budgets = plan_file_budgets(stats, 40 * GiB, depth=2)
    assert budgets == sorted(budgets, reverse=True)
    # first file: (40 - 4) / 2 = 18 GiB; last: (40 - 32) / 2 = 4 GiB
    assert budgets[0] == 18 * GiB
    assert budgets[-1] == 4 * GiB


def test_whole_file_when_ample():
    # budget >> everything: every per-file budget exceeds its span, so
    # plan_chunks would return a single whole-span chunk per file.
    stats = [_st(f"f{i}", 2 * GiB) for i in range(4)]
    budgets = plan_file_budgets(stats, 100 * GiB, depth=2)
    assert all(b >= st.span_bytes for b, st in zip(budgets, stats))


def test_min_with_max_batch_bytes():
    stats = [_st("f0", 1 * GiB)]
    budgets = plan_file_budgets(stats, 100 * GiB, depth=1, max_batch_bytes=512 * MiB)
    assert budgets == [512 * MiB]


def test_group_resident_charges_the_whole_batch_group():
    # Broadcast leaves every rank holding every file of the batch group, so a
    # group's kept bytes go resident together: all its files are charged the
    # group total and share one budget, instead of each file being charged
    # only its own prefix and leaving the rest of its group unbudgeted.
    stats = [
        _st("f0", 8 * GiB, largest=1 * GiB),
        _st("f1", 2 * GiB, largest=1 * GiB),
        _st("f2", 8 * GiB, largest=1 * GiB),
        _st("f3", 2 * GiB, largest=1 * GiB),
    ]
    grouped = plan_file_budgets(stats, 40 * GiB, depth=2, group_size=2)
    assert grouped[0] == grouped[1] and grouped[2] == grouped[3]

    # A group is charged exactly as if it were one file of the group's total
    # kept bytes -- an independent characterization of the rule.
    merged = plan_file_budgets(
        [_st("g0", 10 * GiB, largest=1 * GiB), _st("g1", 10 * GiB, largest=1 * GiB)],
        40 * GiB,
        depth=2,
    )
    assert [grouped[0], grouped[2]] == merged

    # Charging per file instead would hand f0 more than its group can afford.
    per_file = plan_file_budgets(stats, 40 * GiB, depth=2, group_size=1)
    assert per_file[0] > grouped[0]
    assert per_file[1] == grouped[1]  # last file of a group is charged alike


def test_accumulate_resident_false_is_uniform():
    stats = [_st(f"f{i}", 8 * GiB, largest=1 * GiB) for i in range(6)]
    budgets = plan_file_budgets(stats, 12 * GiB, depth=2, accumulate_resident=False)
    assert budgets == [6 * GiB] * 6


def test_yield_clone_reduces_uniform_budget():
    stats = [_st(f"f{i}", 8 * GiB, largest=1 * GiB) for i in range(6)]
    budgets = plan_file_budgets(
        stats,
        12 * GiB,
        depth=2,
        accumulate_resident=False,
        account_for_yield_clone=True,
    )
    assert budgets == [4 * GiB] * 6


def test_infeasible_raises_with_details():
    # 10 files x 2 GiB resident vs 12 GiB budget: infeasible partway through
    stats = [_st(f"f{i}", 2 * GiB, largest=1 * GiB) for i in range(10)]
    with pytest.raises(BudgetInfeasibleError) as ei:
        plan_file_budgets(stats, 12 * GiB, depth=2)
    msg = str(ei.value)
    # Name the file that broke the fit and the bytes it needs -- that number is
    # what a user resizes their budget by, so pin both rather than a substring
    # that the message contains no matter which file failed.
    assert "'f5'" in msg, msg
    assert str(12 * GiB + 2 * GiB) in msg, msg
    assert "queue_size" in msg


def test_nonpositive_budget_and_bad_depth():
    with pytest.raises(BudgetInfeasibleError):
        plan_file_budgets([_st("f", 1)], 0, depth=1)
    # BudgetInfeasibleError subclasses ValueError, so match the message too --
    # otherwise the wrong exception type would satisfy this.
    with pytest.raises(ValueError, match="depth"):
        plan_file_budgets([_st("f", 1)], 1, depth=0)
    with pytest.raises(ValueError, match="group_size"):
        plan_file_budgets([_st("f", 1)], 1, depth=1, group_size=0)


def test_empty_kept_file_contributes_nothing():
    stats = [_st("f0", 4 * GiB), FileWeightStats("f1", 0, 0, 0), _st("f2", 4 * GiB)]
    budgets = plan_file_budgets(stats, 20 * GiB, depth=1)
    # empty file consumes no resident: f2's budget only reflects f0 + f2
    assert budgets[2] == 20 * GiB - 8 * GiB


# ---- simulation property test: replay the plan, assert peak <= budget ----


def _chunk_sizes(span, budget):
    """The chunk spans plan_chunks would produce for a file of `span` bytes."""
    if budget >= span:
        return [span]
    sizes, rem = [], span
    while rem > 0:
        sizes.append(min(budget, rem))
        rem -= min(budget, rem)
    return sizes


def _simulate_peak(
    stats, budgets, base_depth, group_size=1, accumulate_resident=True, multiplier=1
):
    """Replay the load and return the worst peak on any single rank.

    Files are processed in groups of `group_size`, one per rank; chunk-batch j
    carries each rank's j-th chunk (nothing once its file runs out). Broadcast
    leaves every rank holding every tensor, so a batch's bytes become resident
    on every rank once it is consumed. A rank holds up to `base_depth` of its
    own chunk buffers (each costing `multiplier` x its span) plus, under
    broadcast, one in-flight receive tensor.

    Resident is accumulated from the replay itself rather than from the
    planner's R[G(i)+1] formula, so a wrong formula cannot cancel out here.
    """
    batches = []  # list of {rank: (file_idx, size)}
    for start in range(0, len(stats), group_size):
        group = list(range(start, min(start + group_size, len(stats))))
        per_rank = {
            r: _chunk_sizes(stats[i].span_bytes, budgets[i])
            for r, i in enumerate(group)
            if stats[i].kept_bytes > 0
        }
        for j in range(max((len(v) for v in per_rank.values()), default=0)):
            batches.append(
                {r: (group[r], s[j]) for r, s in per_rank.items() if j < len(s)}
            )

    # bytes materialized once batch k has been consumed
    batch_bytes = [sum(sz for _, sz in b.values()) for b in batches]
    peak = 0
    for k in range(len(batches)):
        start = max(0, k - base_depth + 1)
        resident = sum(batch_bytes[:start]) if accumulate_resident else 0
        recv = max((sz for _, sz in batches[k].values()), default=0)
        for r in batches[k]:
            own = sum(b[r][1] for b in batches[start : k + 1] if r in b)
            live = multiplier * own + (recv if group_size > 1 else 0)
            peak = max(peak, resident + live)
    return peak


def test_simulation_peak_within_budget():
    rng = random.Random(1234)
    trials = 400
    multi_chunk = infeasible = grouped = 0
    for trial in range(trials):
        n = rng.randint(1, 12)
        stats = []
        for i in range(n):
            kept = rng.randint(1, 64) * MiB
            largest = max(1, kept // rng.randint(2, 8))
            stats.append(FileWeightStats(f"f{i}", kept, kept, largest))
        qs = rng.choice([-1, 0, 1, 3])
        # The replay's depth model is written out here rather than taken from
        # pipeline_depth(): the planner below calls the real function, so if it
        # ever disagrees with these semantics the planner hands out budgets
        # sized for the wrong number of live buffers and the replay catches it.
        # Sharing one function would let the error cancel on both sides.
        base_depth = 1 if qs < 0 else qs + 2
        group_size = rng.choice([1, 1, 2, 4])
        mult = rng.choice([1, 1, 2])
        depth = pipeline_depth(qs) + (1 if group_size > 1 else 0)
        acc = rng.random() < 0.5
        max_largest = max(s.largest_tensor for s in stats)
        # Headroom is sometimes too small for the largest-tensor floor, so the
        # planner has to refuse rather than hand back an unusable budget.
        budget = (
            sum(s.kept_bytes for s in stats)
            + rng.randint(0, 32) * MiB
            + rng.randint(0, depth * mult) * max_largest
        )
        try:
            budgets = plan_file_budgets(
                stats,
                budget,
                depth,
                accumulate_resident=acc,
                transient_multiplier=mult,
                group_size=group_size,
            )
        except BudgetInfeasibleError:
            infeasible += 1
            continue
        assert all(b >= st.largest_tensor for b, st in zip(budgets, stats))
        if any(
            len(_chunk_sizes(st.span_bytes, b)) >= 3 for st, b in zip(stats, budgets)
        ):
            multi_chunk += 1
        if group_size > 1 and n > group_size:
            grouped += 1
        peak = _simulate_peak(
            stats,
            budgets,
            base_depth,
            group_size=group_size,
            accumulate_resident=acc,
            multiplier=mult,
        )
        # acc=True: resident + transient <= budget. acc=False: the plan bounds
        # the transient side only (destinations preallocated), and the sim
        # models exactly that side -> same assertion.
        assert peak <= budget, (trial, peak, budget, acc, group_size, mult, qs)
    # Guard against the corpus quietly degenerating: the bound says nothing
    # where nothing is split, the infeasibility floor is untested if no plan is
    # ever refused, and the group-resident rule is untested without multi-rank
    # groups. Failing these means the generator drifted, not the planner.
    assert multi_chunk > 40, f"only {multi_chunk}/{trials} trials split a file 3+ ways"
    assert infeasible > 20, f"only {infeasible}/{trials} trials hit the budget floor"
    assert grouped > 40, f"only {grouped}/{trials} trials used multi-rank groups"


# ---- collect_file_stats on real files ----


def test_collect_file_stats_matches_headers(input_files, framework):
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    (st,) = collect_file_stats([(input_files[0], meta)])
    total = sum(f.data_offsets[1] - f.data_offsets[0] for f in meta.tensors.values())
    largest = max(f.data_offsets[1] - f.data_offsets[0] for f in meta.tensors.values())
    assert st.kept_bytes == total
    assert st.largest_tensor == largest
    assert st.span_bytes >= largest

    # filtered: keep every other tensor -> kept < total, span may include holes
    names = sorted(meta.tensors.keys())
    keep = set(names[::2])
    (fst,) = collect_file_stats([(input_files[0], meta)], lambda n: n in keep)
    assert 0 < fst.kept_bytes < total
    assert fst.span_bytes >= fst.kept_bytes


# ---- end-to-end on CPU: budgeted load is byte-identical to a plain load ----


def test_parallel_loader_device_memory_budget_cpu(input_files, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only integration test")
    import torch
    from safetensors.torch import load_file

    from fastsafetensors import ParallelLoader

    expected = load_file(input_files[0])
    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    (st,) = collect_file_stats([(input_files[0], meta)])
    # tight budget: forces chunking (several chunks) but stays feasible
    budget = st.kept_bytes + pipeline_depth(0) * st.largest_tensor + 4096

    pl = ParallelLoader(
        pg=None,
        hf_weights_files=[input_files[0]],
        device="cpu",
        nogds=True,
        use_tqdm_on_load=False,
        device_memory_budget=budget,
    )
    got = dict(pl.iterate_weights())
    assert set(got.keys()) == set(expected.keys())
    for k in expected:
        assert torch.equal(got[k], expected[k]), k


@pytest.mark.parametrize(
    ("accumulate_resident", "expected_account_for_clone"),
    [(True, False), (False, True)],
)
def test_single_process_loader_budgets_yield_clone(
    input_files,
    framework,
    monkeypatch,
    accumulate_resident,
    expected_account_for_clone,
):
    if framework.get_name() != "pytorch":
        pytest.skip("pytorch-only integration test")

    from fastsafetensors import ParallelLoader, _planner

    captured = {}
    real_plan_file_budgets = _planner.plan_file_budgets

    def record_plan(*args, **kwargs):
        captured.update(kwargs)
        return real_plan_file_budgets(*args, **kwargs)

    monkeypatch.setattr(_planner, "plan_file_budgets", record_plan)
    pl = ParallelLoader(
        pg=None,
        hf_weights_files=[input_files[0]],
        device="cpu",
        nogds=True,
        use_tqdm_on_load=False,
        device_memory_budget=1 << 30,
        accumulate_resident=accumulate_resident,
    )
    try:
        assert pl.need_clone
        assert captured["account_for_yield_clone"] is expected_account_for_clone
    finally:
        pl.close()


def test_transient_multiplier_infeasible():
    # a plan feasible at 1x must fail at 2x when the budget is tight
    stats = [_st("f0", 2 * GiB)]
    budget = stats[0].kept_bytes + 2 * stats[0].largest_tensor  # depth=2, k=1 fits
    plan_file_budgets(stats, budget, depth=2, transient_multiplier=1)
    with pytest.raises(BudgetInfeasibleError):
        plan_file_budgets(stats, budget, depth=2, transient_multiplier=2)


def test_transient_multiplier_validation():
    with pytest.raises(ValueError):
        plan_file_budgets([_st("f0", GiB)], GiB, depth=1, transient_multiplier=0)
