"""Tests for histo_miner.tissue_analyser.

Covers the functions that produce the published feature set: cell counting,
per-class counts inside the tumour mask, cell-to-cell distances, and the
`hvn_outputproperties` aggregation that assembles the final feature dictionary.

Two groups are skipped at the bottom:
  * `cell2celldist_classjson` -- the single-process reference implementation.
    The pipeline calls the multiprocessing variant, which is tested above.
  * morphology -- `calculate_morphologies` is False in
    `configs/histo_miner_pipeline.yml` and no morphology feature appears in the
    released feature ranking (`data/feature_rank/Ranking_of_features.json`).

Geometry note: every synthetic cell is an axis-aligned square, so its area is
exactly side**2, its circularity 4*pi*s**2/(4s)**2 = pi/4, and its aspect ratio
1.0 -- all hand-checkable without running the code.
"""
import json
import math

import numpy as np
import pytest
from PIL import Image

from histo_miner.tissue_analyser import (
    cell2celldist_classjson,
    cells_insidemask_classjson,
    count_pix_value,
    counthvnjson,
    countjson,
    hvn_outputproperties,
    morph_classandmargin_classjson,
    mpcell2celldist_classjson,
)

MASK_SIZE = 100
MASK_LO, MASK_HI = 20, 80
TUMOUR_MARGIN = 10

# Class 1: three squares of side 4, 6, 8 -> areas 16, 36, 64
CLASS1_HALF_WIDTHS = (2, 3, 4)
CLASS1_AREAS = [(2 * h) ** 2 for h in CLASS1_HALF_WIDTHS]
# Class 2: two squares of side 4 -> areas 16, 16
CLASS2_AREAS = [16, 16]

CLASS_NAMES = ["Granulocyte", "Lymphocyte", "Plasma", "Stroma", "Tumor"]

# Index of each statistic inside the 21-value per-class morphology vector.
AREAS_MEAN, AREAS_STD, AREAS_MEDIAN = 0, 1, 2
CIRC_MEAN, CIRC_STD = 7, 8
ASPECT_MEAN, ASPECT_STD = 14, 15


def _square_contour(cx: int, cy: int, half: int) -> list:
    """Axis-aligned square polygon centred on (cx, cy)."""
    return [
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ]


@pytest.fixture
def maskmap_path(tmp_path):
    """100x100 binary mask with a solid tumour region spanning rows/cols 20-79."""
    mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)
    mask[MASK_LO:MASK_HI, MASK_LO:MASK_HI] = 255
    path = tmp_path / "maskmap.png"
    Image.fromarray(mask).save(path)
    return str(path)


@pytest.fixture
def classjson_path(tmp_path):
    """Three cell classes with known positions.

    class 1: three cells inside the tumour, on its upper border (y = 20/21).
             That band is also where dilate(mask) - erode(mask) is non-zero for a
             5x5 kernel, so they count for the vicinity too.
    class 2: two cells inside the tumour, well away from class 1.
    class 3: two cells far outside the tumour -- the "absent class" case.
    """
    nuclei = {}
    # centroid is [x, y]; the maskmap is indexed [y, x]
    for i, ((cx, cy), half) in enumerate(
        zip([(40, 20), (50, 21), (60, 20)], CLASS1_HALF_WIDTHS)
    ):
        nuclei[str(i)] = {"centroid": [cx, cy], "type": 1,
                          "contour": _square_contour(cx, cy, half)}
    for i, (cx, cy) in enumerate([(30, 50), (70, 50)], start=10):
        nuclei[str(i)] = {"centroid": [cx, cy], "type": 2,
                          "contour": _square_contour(cx, cy, 2)}
    for i, (cx, cy, half) in enumerate([(5, 5, 2), (9, 9, 3)], start=20):
        nuclei[str(i)] = {"centroid": [cx, cy], "type": 3,
                          "contour": _square_contour(cx, cy, half)}

    path = tmp_path / "cells.json"
    path.write_text(json.dumps(nuclei))
    return str(path)


# --------------------------------------------------------------------------- #
# count_pix_value -- feeds the tumour area that every density feature divides by
# --------------------------------------------------------------------------- #

def test_count_pix_value_counts_the_mask_area(maskmap_path):
    assert count_pix_value(maskmap_path, 255) == (MASK_HI - MASK_LO) ** 2


def test_count_pix_value_counts_the_background(maskmap_path):
    assert count_pix_value(maskmap_path, 0) == MASK_SIZE**2 - (MASK_HI - MASK_LO) ** 2


def test_count_pix_value_returns_zero_for_an_absent_value(maskmap_path):
    assert count_pix_value(maskmap_path, 7) == 0


# --------------------------------------------------------------------------- #
# countjson / counthvnjson
#
# Both count raw substring occurrences in the file text, not parsed JSON values.
# --------------------------------------------------------------------------- #

def test_countjson_counts_substring_occurrences(tmp_path):
    path = tmp_path / "cells.json"
    path.write_text('{"0": "Tumor", "1": "Tumor", "2": "Stroma"}')
    assert countjson(str(path), ["Tumor", "Stroma", "Lymphocyte"]) == {
        "Tumor": 2, "Stroma": 1, "Lymphocyte": 0,
    }


def test_countjson_matches_partial_words(tmp_path):
    """Documented footgun: a plain substring count, so a shorter word also
    matches inside a longer one."""
    path = tmp_path / "cells.json"
    path.write_text('{"0": "Tumor", "1": "TumorMargin"}')
    assert countjson(str(path), ["Tumor"]) == {"Tumor": 2}


def test_counthvnjson_without_names_behaves_like_countjson(tmp_path):
    path = tmp_path / "cells.json"
    path.write_text('{"0": {"type": 1}, "1": {"type": 1}, "2": {"type": 2}}')
    assert counthvnjson(str(path), ['"type": 1', '"type": 2']) == {
        '"type": 1': 2, '"type": 2': 1,
    }


def test_counthvnjson_prepends_background_to_the_class_names(tmp_path):
    """classnameaskey is zipped against ['Background'] + names, so the first
    searched word is always reported as the background class."""
    path = tmp_path / "cells.json"
    path.write_text('{"0": {"type": 0}, "1": {"type": 1}, "2": {"type": 2}}')
    assert counthvnjson(
        str(path), ['"type": 0', '"type": 1', '"type": 2'],
        classnameaskey=["Granulocyte", "Lymphocyte"],
    ) == {"Background": 1, "Granulocyte": 1, "Lymphocyte": 1}


# --------------------------------------------------------------------------- #
# cells_insidemask_classjson -- per-class counts feeding the published ratios
# --------------------------------------------------------------------------- #

def test_counts_only_cells_inside_the_mask(maskmap_path, classjson_path):
    """Classes 1 and 2 are inside the tumour; class 3 is outside."""
    result = cells_insidemask_classjson(
        maskmap=maskmap_path, classjson=classjson_path,
        selectedclasses=[1, 2, 3], maskmapdownfactor=1,
    )
    assert list(result["list_numinstanceperclass"]) == [3, 2, 0]


def test_sums_instance_areas_per_class(maskmap_path, classjson_path):
    """Class 1: 16 + 36 + 64 = 116. Class 2: 16 + 16 = 32."""
    result = cells_insidemask_classjson(
        maskmap=maskmap_path, classjson=classjson_path,
        selectedclasses=[1, 2, 3], maskmapdownfactor=1,
    )
    assert list(result["list_totareainstanceperclass"]) == [
        sum(CLASS1_AREAS), sum(CLASS2_AREAS), 0,
    ]


def test_classnameaskey_maps_class_numbers_to_names(maskmap_path, classjson_path):
    """Names are indexed by class number minus one."""
    result = cells_insidemask_classjson(
        maskmap=maskmap_path, classjson=classjson_path,
        selectedclasses=[1, 2, 3], maskmapdownfactor=1, classnameaskey=CLASS_NAMES,
    )
    assert result["dict_numinstanceperclass"] == {
        "Granulocyte": 3, "Lymphocyte": 2, "Plasma": 0,
    }
    assert result["dict_totareainstanceperclass"]["Granulocyte"] == sum(CLASS1_AREAS)


def test_rejects_a_four_channel_maskmap(tmp_path, classjson_path):
    rgba = np.zeros((MASK_SIZE, MASK_SIZE, 4), dtype=np.uint8)
    path = tmp_path / "rgba.png"
    Image.fromarray(rgba, mode="RGBA").save(path)
    with pytest.raises(ValueError):
        cells_insidemask_classjson(
            maskmap=str(path), classjson=classjson_path, selectedclasses=[1],
        )


# --------------------------------------------------------------------------- #
# hvn_outputproperties -- assembles the published feature dictionary
# --------------------------------------------------------------------------- #

WSI_COUNTS = {
    "Background": 10, "Granulocyte": 10, "Lymphocyte": 20,
    "Plasma": 30, "Stroma": 40, "Tumor": 100, "Epithelial": 0,
}
TOTAL_CELLS = 200  # sum of the above minus Background
EPS = 0.001        # matches the epsilon used in the source

SECTIONS = {
    "CalculationsforWSI",
    "CalculationsRatiosinsideTumor",
    "CalculationsMorphinsideTumor",
    "CalculationsDistinsideTumor",
    "CalculationsMixed",
}


def test_result_has_the_five_top_level_sections():
    assert set(hvn_outputproperties(allcells_in_wsi_dict=dict(WSI_COUNTS))) == SECTIONS


def test_called_with_no_arguments_returns_empty_sections():
    """Every input is optional; absent inputs must not raise."""
    assert set(hvn_outputproperties()) == SECTIONS


def test_wsi_percentages_exclude_background_from_the_total():
    pct = hvn_outputproperties(allcells_in_wsi_dict=dict(WSI_COUNTS))[
        "CalculationsforWSI"]["Percentages_of_cell_types_in_WSI"]
    assert pct["Granulocytes_Percentage"] == pytest.approx(10 / TOTAL_CELLS)
    assert pct["Lymphocytes_Percentage"] == pytest.approx(20 / TOTAL_CELLS)
    assert pct["TumorCells_Percentage"] == pytest.approx(100 / TOTAL_CELLS)
    assert pct["EpithelialCells_Percentage"] == pytest.approx(0.0)


def test_wsi_percentages_sum_to_one():
    pct = hvn_outputproperties(allcells_in_wsi_dict=dict(WSI_COUNTS))[
        "CalculationsforWSI"]["Percentages_of_cell_types_in_WSI"]
    assert sum(pct.values()) == pytest.approx(1.0)


def test_wsi_log_ratio_matches_the_epsilon_smoothed_formula():
    """log((a + eps) / (b + eps)) with eps = 0.001."""
    ratios = hvn_outputproperties(allcells_in_wsi_dict=dict(WSI_COUNTS))[
        "CalculationsforWSI"]["Ratios_between_cell_types_WSI"]
    assert ratios["LogRatio_Granulocytes_TumorCells"] == pytest.approx(
        math.log((10 + EPS) / (100 + EPS))
    )


def test_log_ratio_of_an_absent_class_is_finite():
    """The epsilon exists so that a zero count does not produce -inf."""
    ratios = hvn_outputproperties(allcells_in_wsi_dict=dict(WSI_COUNTS))[
        "CalculationsforWSI"]["Ratios_between_cell_types_WSI"]
    value = ratios["LogRatio_EpithelialCells_TumorCells"]
    assert math.isfinite(value)
    assert value == pytest.approx(math.log(EPS / (100 + EPS)))


# --------------------------------------------------------------------------- #
# mpcell2celldist_classjson -- the variant the pipeline actually calls
# (scripts/main4_tissue_analyser.py uses cellfilter='Tumor' with a maskmap)
#
# Returns a nested list, one dict of distribution statistics per class pair:
#   [[{'dist_mean': ..., 'dist_std': ..., 'dist_median': ..., 'dist_MAD': ...,
#      'dist_skewness': ..., 'dist_kurt': ..., 'dist_iqr': ...}]]
# --------------------------------------------------------------------------- #

# Closest class-2 neighbour for each class-1 cell:
#   (40,20) -> (30,50) : hypot(10, 30)
#   (50,21) -> either  : hypot(20, 29)
#   (60,20) -> (70,50) : hypot(10, 30)
D_NEAR = math.hypot(10, 30)   # 31.6228
D_FAR = math.hypot(20, 29)    # 35.2278
MIN_DISTANCES = [D_NEAR, D_FAR, D_NEAR]

DIST_KEYS = {"dist_mean", "dist_std", "dist_median",
             "dist_MAD", "dist_skewness", "dist_kurt", "dist_iqr"}


def _mp_distances(maskmap_path, classjson_path, selectedclasses):
    return mpcell2celldist_classjson(
        classjson_path,
        selectedclasses,
        cellfilter="Tumor",
        maskmap=maskmap_path,
        maskmapdownfactor=1,
        tumormargin=None,
    )


def test_mp_distances_return_one_stats_dict_per_class_pair(maskmap_path, classjson_path):
    """Two classes -> one pair -> one dict of seven distribution statistics."""
    result = _mp_distances(maskmap_path, classjson_path, [1, 2])
    assert isinstance(result, list)
    assert len(result) == 1 and len(result[0]) == 1
    assert set(result[0][0]) == DIST_KEYS


def test_mp_distance_statistics_match_the_synthetic_cell_positions(
    maskmap_path, classjson_path
):
    """Each class-1 cell's closest class-2 neighbour is hypot(10,30) or
    hypot(20,29); the reported statistics must describe exactly those three."""
    stats = _mp_distances(maskmap_path, classjson_path, [1, 2])[0][0]
    assert stats["dist_mean"] == pytest.approx(np.mean(MIN_DISTANCES))
    assert stats["dist_median"] == pytest.approx(np.median(MIN_DISTANCES))
    assert stats["dist_std"] == pytest.approx(np.std(MIN_DISTANCES))
    assert stats["dist_MAD"] == pytest.approx(
        np.mean(np.abs(np.asarray(MIN_DISTANCES) - np.mean(MIN_DISTANCES)))
    )


# --------------------------------------------------------------------------- #
# Skipped: single-process distance variant
#
# `cell2celldist_classjson` is the reference implementation; the pipeline calls
# the multiprocessing variant tested above.
# --------------------------------------------------------------------------- #

@pytest.mark.skip(reason="Single-process reference implementation; the pipeline "
                         "uses mpcell2celldist_classjson, tested above.")
def test_single_process_distances_agree_with_the_multiprocess_variant(
    maskmap_path, classjson_path
):
    single = cell2celldist_classjson(
        classjson_path, [1, 2], cellfilter="Tumor",
        maskmap=maskmap_path, maskmapdownfactor=1, tumormargin=None,
    )
    multi = _mp_distances(maskmap_path, classjson_path, [1, 2])
    assert single[0][0] == pytest.approx(multi[0][0])


# --------------------------------------------------------------------------- #
# Skipped: morphology
#
# Both per-class loops contain an `if len(areas_class) == 0:` branch that assigns
# areas_mean / circularities_mean / aspectratios_mean, while the corresponding
# `else` branch and the consumer use areas_vic_mean / areas_mask_mean. The zero
# assignments are therefore never read: a class absent from the region raises
# UnboundLocalError if it is the first one processed, or otherwise keeps the
# previously processed class's values. The runtime warning is the safeguard.
# --------------------------------------------------------------------------- #

SKIP_MORPH = pytest.mark.skip(
    reason="Morphology is disabled by default (calculate_morphologies: False) and "
           "is not part of the published feature set."
)


def _morph(maskmap_path, classjson_path, tum, vic):
    return morph_classandmargin_classjson(
        maskmap=maskmap_path, classjson=classjson_path,
        selectedclassestum=tum, selectedclassesvic=vic,
        maskmapdownfactor=1, tumormargin=TUMOUR_MARGIN,
    )


@SKIP_MORPH
def test_morph_returns_21_values_per_class(maskmap_path, classjson_path):
    tumour = _morph(maskmap_path, classjson_path, [1], [1])["list_morphologyfeatperclass"]
    assert len(tumour) == 1
    assert len(tumour[0]) == 21


@SKIP_MORPH
def test_morph_area_statistics_match_the_synthetic_squares(maskmap_path, classjson_path):
    feats = _morph(maskmap_path, classjson_path, [1], [1])["list_morphologyfeatperclass"][0]
    assert feats[AREAS_MEAN] == pytest.approx(np.mean(CLASS1_AREAS))
    assert feats[AREAS_MEDIAN] == pytest.approx(np.median(CLASS1_AREAS))
    assert feats[AREAS_STD] == pytest.approx(np.std(CLASS1_AREAS))


@SKIP_MORPH
def test_morph_circularity_of_a_square_is_pi_over_four(maskmap_path, classjson_path):
    feats = _morph(maskmap_path, classjson_path, [1], [1])["list_morphologyfeatperclass"][0]
    assert feats[CIRC_MEAN] == pytest.approx(math.pi / 4)
    assert feats[CIRC_STD] == pytest.approx(0.0, abs=1e-12)


@SKIP_MORPH
def test_morph_aspect_ratio_of_a_square_is_one(maskmap_path, classjson_path):
    feats = _morph(maskmap_path, classjson_path, [1], [1])["list_morphologyfeatperclass"][0]
    assert feats[ASPECT_MEAN] == pytest.approx(1.0)
    assert feats[ASPECT_STD] == pytest.approx(0.0, abs=1e-12)


@SKIP_MORPH
def test_morph_class_absent_from_tumour_yields_zero_features(maskmap_path, classjson_path):
    """States intended behaviour; not implemented, see the note above."""
    tumour = _morph(maskmap_path, classjson_path, [1, 3], [1])["list_morphologyfeatperclass"]
    assert all(value == 0 for value in tumour[1])
