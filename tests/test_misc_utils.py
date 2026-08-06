"""Unit tests for the pure helper functions in histo_miner.utils.misc."""
import json

import numpy as np
import pytest

from histo_miner.utils.misc import (
    NpEncoder,
    convert_flatten,
    convert_flatten_redundant,
    convert_names_to_integers,
    convert_names_to_orderedint,
    find_closest_sublist,
    get_indices_by_value,
    noheadercsv_to_dict,
    rename_with_ancestors,
    rename_with_parent,
)


# --------------------------------------------------------------------------- #
# convert_flatten_redundant
# --------------------------------------------------------------------------- #

def test_flatten_redundant_joins_nested_keys_with_underscore():
    nested = {"A": {"B": {"c": 1, "d": 2}}}
    assert convert_flatten_redundant(nested) == {"A_B_c": 1, "A_B_d": 2}


def test_flatten_redundant_keeps_same_leaf_name_under_different_parents():
    nested = {"P1": {"x": 1}, "P2": {"x": 2}}
    assert convert_flatten_redundant(nested) == {"P1_x": 1, "P2_x": 2}


def test_flatten_redundant_drops_empty_dicts():
    assert convert_flatten_redundant({"A": {"b": 1}, "E": {}}) == {"A_b": 1}


def test_flatten_redundant_honours_custom_separator():
    assert convert_flatten_redundant({"A": {"b": 1}}, sep=".") == {"A.b": 1}


# --------------------------------------------------------------------------- #
# convert_flatten
# --------------------------------------------------------------------------- #

def test_flatten_keeps_only_leaf_key_names():
    nested = {"A": {"B": {"c": 1, "d": 2}}}
    assert convert_flatten(nested) == {"c": 1, "d": 2}


def test_flatten_leaf_name_collisions_overwrite_silently():
    """Documented footgun: identical leaf names collapse, last one wins."""
    nested = {"P1": {"x": 1}, "P2": {"x": 2}}
    assert convert_flatten(nested) == {"x": 2}


# --------------------------------------------------------------------------- #
# rename_with_parent / rename_with_ancestors
# --------------------------------------------------------------------------- #

NESTED = {
    "level1": {
        "level2a": {"key1": "value1", "key_to_rename": "value2"},
        "level2b": {"key2": "value3"},
    }
}


def test_rename_with_parent_prefixes_immediate_parent():
    out = rename_with_parent(NESTED, ["key_to_rename"])
    assert out["level1"]["level2a"] == {
        "key1": "value1",
        "level2a_key_to_rename": "value2",
    }
    assert out["level1"]["level2b"] == {"key2": "value3"}


def test_rename_with_ancestors_prefixes_parent_and_grandparent():
    out = rename_with_ancestors(NESTED, ["key_to_rename"])
    assert out["level1"]["level2a"] == {
        "key1": "value1",
        "level1_level2a_key_to_rename": "value2",
    }


def test_rename_leaves_input_untouched():
    rename_with_parent(NESTED, ["key_to_rename"])
    assert "key_to_rename" in NESTED["level1"]["level2a"]


# --------------------------------------------------------------------------- #
# name -> integer conversions
# --------------------------------------------------------------------------- #

def test_orderedint_assigns_sequential_ids_in_first_seen_order():
    assert convert_names_to_orderedint(["a", "b", "a", "c"]) == [1, 2, 1, 3]


def test_orderedint_is_deterministic_across_calls():
    names = ["p3", "p1", "p3", "p2"]
    assert convert_names_to_orderedint(names) == convert_names_to_orderedint(names)


def test_names_to_integers_gives_each_distinct_name_a_distinct_id():
    """Only distinctness and stability *within a run* are required: the ids are
    regenerated from the patient names on every execution and never persisted."""
    out = convert_names_to_integers(["a", "b", "a"])
    assert out[0] == out[2]
    assert out[0] != out[1]
    assert len(out) == 3

# --------------------------------------------------------------------------- #
# small list helpers
# --------------------------------------------------------------------------- #

def test_get_indices_by_value_groups_positions():
    assert get_indices_by_value(["a", "b", "a"]) == {"a": [0, 2], "b": [1]}


def test_find_closest_sublist_picks_nearest_length():
    assert find_closest_sublist([[1], [1, 2, 3], [1, 2]], 2) == [1, 2]


def test_find_closest_sublist_breaks_ties_with_the_first_match():
    assert find_closest_sublist([[1, 2], [3, 4]], 2) == [1, 2]


# --------------------------------------------------------------------------- #
# noheadercsv_to_dict
# --------------------------------------------------------------------------- #

def test_noheadercsv_maps_first_column_to_second(tmp_path):
    csv_file = tmp_path / "pairs.csv"
    csv_file.write_text("sample_a,1\nsample_b,2\nsample_c,3\n")
    assert noheadercsv_to_dict(str(csv_file)) == {
        "sample_a": "1",
        "sample_b": "2",
        "sample_c": "3",
    }


# --------------------------------------------------------------------------- #
# NpEncoder  -- every feature JSON is written through this
# --------------------------------------------------------------------------- #

def test_npencoder_serialises_numpy_scalars_and_arrays():
    payload = {"i": np.int64(3), "f": np.float32(1.5), "arr": np.arange(3)}
    assert json.loads(json.dumps(payload, cls=NpEncoder)) == {
        "i": 3,
        "f": 1.5,
        "arr": [0, 1, 2],
    }


def test_npencoder_rejects_numpy_bool():
    """np.bool_ is neither np.integer nor np.floating, so it reaches the
    fallback and raises rather than being silently mis-encoded."""
    with pytest.raises(TypeError):
        json.dumps({"flag": np.bool_(True)}, cls=NpEncoder)