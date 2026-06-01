"""
Tests for obf_psychiatric_pipeline.cv.folds.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from obf_psychiatric_pipeline.cv.folds import (
    generate_repeated_folds,
    iter_folds,
    load_folds,
    save_folds,
    validate_folds,
)


@pytest.fixture()
def small_ids() -> list[str]:
    return [f"p{i:02d}" for i in range(10)]


@pytest.fixture()
def small_folds(small_ids) -> dict:
    return generate_repeated_folds(
        participant_ids=small_ids, n_splits=5, n_reps=3, seeds=[0, 1, 2]
    )


class TestGenerateRepeatedFolds:
    def test_metadata_fields(self, small_folds, small_ids):
        m = small_folds["metadata"]
        assert m["n_splits"] == 5
        assert m["n_reps"] == 3
        assert m["seeds"] == [0, 1, 2]
        assert m["n_participants"] == len(small_ids)
        assert sorted(m["participant_ids"]) == sorted(small_ids)
        assert "generated_at" in m
        assert "platform" in m

    def test_repetition_count(self, small_folds):
        assert len(small_folds["repetitions"]) == 3

    def test_fold_count_per_rep(self, small_folds):
        for rep in small_folds["repetitions"]:
            assert len(rep["folds"]) == 5

    def test_each_participant_in_test_exactly_once_per_rep(self, small_folds, small_ids):
        for rep_idx, rep in enumerate(small_folds["repetitions"]):
            counts: Counter = Counter()
            for fold in rep["folds"]:
                counts.update(fold["test"])
            for pid in small_ids:
                assert counts[pid] == 1, (
                    f"rep {rep_idx}: participant {pid} appeared {counts[pid]} times"
                )

    def test_each_participant_in_test_n_reps_times_across_all_reps(self, small_folds, small_ids):
        n_reps = small_folds["metadata"]["n_reps"]
        counts: Counter = Counter()
        for rep in small_folds["repetitions"]:
            for fold in rep["folds"]:
                counts.update(fold["test"])
        for pid in small_ids:
            assert counts[pid] == n_reps

    def test_test_sets_are_disjoint_within_rep(self, small_folds):
        for rep_idx, rep in enumerate(small_folds["repetitions"]):
            seen: set = set()
            for fold_idx, fold in enumerate(rep["folds"]):
                test_set = set(fold["test"])
                overlap = seen & test_set
                assert not overlap, f"rep {rep_idx} fold {fold_idx}: overlap {overlap}"
                seen |= test_set

    def test_test_sets_cover_all_participants(self, small_folds, small_ids):
        for rep_idx, rep in enumerate(small_folds["repetitions"]):
            union: set = set()
            for fold in rep["folds"]:
                union.update(fold["test"])
            assert union == set(small_ids)

    def test_different_reps_have_different_fold_assignments(self, small_folds):
        fold_0_tests = [set(f["test"]) for f in small_folds["repetitions"][0]["folds"]]
        fold_1_tests = [set(f["test"]) for f in small_folds["repetitions"][1]["folds"]]
        assert fold_0_tests != fold_1_tests

    def test_determinism_same_seed(self, small_ids):
        folds_a = generate_repeated_folds(small_ids, n_splits=5, n_reps=3, seeds=[7, 8, 9])
        folds_b = generate_repeated_folds(small_ids, n_splits=5, n_reps=3, seeds=[7, 8, 9])
        for rep_a, rep_b in zip(folds_a["repetitions"], folds_b["repetitions"]):
            for fold_a, fold_b in zip(rep_a["folds"], rep_b["folds"]):
                assert fold_a["test"] == fold_b["test"]

    def test_raises_on_seeds_length_mismatch(self, small_ids):
        with pytest.raises(ValueError, match="len\\(seeds\\)"):
            generate_repeated_folds(small_ids, n_reps=3, seeds=[0, 1])

    def test_raises_on_too_few_participants(self):
        with pytest.raises(ValueError, match="n_participants"):
            generate_repeated_folds(["p0", "p1", "p2"], n_splits=5, n_reps=1, seeds=[0])

    def test_raises_on_duplicate_ids(self):
        with pytest.raises(ValueError, match="duplicates"):
            generate_repeated_folds(["p0", "p0", "p1", "p2", "p3"], n_splits=2, n_reps=1)

    def test_seeds_stored_as_ints(self, small_folds):
        for seed in small_folds["metadata"]["seeds"]:
            assert isinstance(seed, int)

    def test_test_ids_sorted(self, small_folds):
        for rep in small_folds["repetitions"]:
            for fold in rep["folds"]:
                assert fold["test"] == sorted(fold["test"])


class TestSaveLoadFolds:
    def test_round_trip(self, small_folds, tmp_path):
        path = tmp_path / "folds.json"
        save_folds(small_folds, path)
        loaded = load_folds(path)
        assert loaded["metadata"] == small_folds["metadata"]
        for rep_a, rep_b in zip(small_folds["repetitions"], loaded["repetitions"]):
            assert rep_a["seed"] == rep_b["seed"]
            for fa, fb in zip(rep_a["folds"], rep_b["folds"]):
                assert fa["test"] == fb["test"]

    def test_saves_valid_json(self, small_folds, tmp_path):
        path = tmp_path / "folds.json"
        save_folds(small_folds, path)
        with open(path) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "repetitions" in data

    def test_creates_parent_directory(self, small_folds, tmp_path):
        path = tmp_path / "nested" / "dir" / "folds.json"
        save_folds(small_folds, path)
        assert path.exists()

    def test_load_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="generate_fold_fixtures"):
            load_folds(tmp_path / "nonexistent.json")


class TestValidateFolds:
    def test_exact_match_passes(self, small_folds, small_ids):
        validate_folds(small_folds, small_ids)

    def test_order_independent(self, small_folds, small_ids):
        validate_folds(small_folds, list(reversed(small_ids)))

    def test_extra_participant_raises(self, small_folds, small_ids):
        with pytest.raises(ValueError, match="not in fixture"):
            validate_folds(small_folds, small_ids + ["extra_participant"])

    def test_missing_participant_raises(self, small_folds, small_ids):
        with pytest.raises(ValueError, match="not in data"):
            validate_folds(small_folds, small_ids[:-1])

    def test_empty_list_raises(self, small_folds):
        with pytest.raises(ValueError):
            validate_folds(small_folds, [])


class TestIterFolds:
    def test_yields_correct_tuple_count(self, small_folds, small_ids):
        ids = np.array(small_ids)
        recs = list(iter_folds(small_folds, ids))
        assert len(recs) == 3 * 5

    def test_masks_are_complementary(self, small_folds, small_ids):
        ids = np.array(small_ids)
        for _, _, train, test in iter_folds(small_folds, ids):
            assert np.all(train ^ test)

    def test_masks_cover_all_participants(self, small_folds, small_ids):
        ids = np.array(small_ids)
        for _, _, train, test in iter_folds(small_folds, ids):
            assert np.sum(train) + np.sum(test) == len(ids)

    def test_rep_and_fold_indices_are_ordered(self, small_folds, small_ids):
        ids = np.array(small_ids)
        reps_seen = [(r, f) for r, f, _, _ in iter_folds(small_folds, ids)]
        expected = [(r, f) for r in range(3) for f in range(5)]
        assert reps_seen == expected

    def test_test_sizes_approximately_equal(self, small_folds, small_ids):
        ids = np.array(small_ids)
        for _, _, _, test in iter_folds(small_folds, ids):
            assert 1 <= np.sum(test) <= 4
