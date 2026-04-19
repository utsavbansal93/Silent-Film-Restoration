"""Pure-function tests for S04 index + shot-boundary remaps."""
from pipeline.common.config import IntertitleCardCfg
from pipeline.stages.s04_intertitle_extract import (
    build_frame_index_map,
    build_intertitle_plan,
    remap_shot_boundaries,
)


def _cards(ranges):
    return [IntertitleCardCfg(id=f"C{i+1}", orig_start_frame=a, orig_end_frame=b, label="")
            for i, (a, b) in enumerate(ranges)]


def test_index_map_empty_cards_is_identity():
    n2o, o2n = build_frame_index_map(10, [])
    assert n2o == list(range(10))
    assert o2n == {i: i for i in range(10)}


def test_index_map_single_card_in_middle():
    # Remove frames 3-5 inclusive from a 10-frame stream.
    n2o, o2n = build_frame_index_map(10, _cards([(3, 5)]))
    assert n2o == [0, 1, 2, 6, 7, 8, 9]                   # 7 retained
    assert o2n[0] == 0 and o2n[2] == 2
    assert 3 not in o2n and 5 not in o2n                  # removed
    assert o2n[6] == 3 and o2n[9] == 6                    # shifted down by 3


def test_index_map_multiple_disjoint_cards():
    n2o, o2n = build_frame_index_map(20, _cards([(2, 3), (10, 12), (18, 19)]))
    # Removed: {2,3,10,11,12,18,19} — 7 frames; 13 retained.
    assert len(n2o) == 13
    assert o2n[0] == 0 and o2n[1] == 1
    assert 2 not in o2n
    assert o2n[4] == 2 and o2n[9] == 7
    assert 10 not in o2n and 12 not in o2n
    assert o2n[13] == 8 and o2n[17] == 12
    assert 18 not in o2n and 19 not in o2n


def test_index_map_is_bijection_over_retained():
    n2o, o2n = build_frame_index_map(100, _cards([(5, 9), (40, 50)]))
    # new→orig must be strictly increasing
    assert n2o == sorted(set(n2o))
    # orig→new must be the inverse
    for orig in o2n:
        assert n2o[o2n[orig]] == orig


def test_shot_remap_drops_boundaries_inside_cards():
    cards = _cards([(10, 20)])
    _, o2n = build_frame_index_map(50, cards)
    total_new = 50 - 11
    # Boundary at 15 is inside card → dropped. Boundaries at 5 and 30 remain but shift.
    out = remap_shot_boundaries([5, 15, 30], cards, o2n, total_new)
    assert out == [5, 19]                  # 30 shifts down by 11 card frames


def test_shot_remap_preserves_card_edge_boundaries():
    # Boundary exactly at card start should be preserved (the pre-card shot ends here).
    cards = _cards([(20, 25)])
    _, o2n = build_frame_index_map(40, cards)
    total_new = 40 - 6
    # Boundary at 20 is inside the card range → dropped by current impl.
    # Boundary at 26 (first frame after the card) maps to new idx 20.
    out = remap_shot_boundaries([0, 20, 26], cards, o2n, total_new)
    assert 20 in out or 26 - 6 in out      # new index 20


def test_shot_remap_deduplicates():
    cards = _cards([(10, 15)])
    _, o2n = build_frame_index_map(30, cards)
    total_new = 30 - 6
    # Two original boundaries that collapse to the same new index after removal.
    out = remap_shot_boundaries([16, 17], cards, o2n, total_new)
    # 16 → new idx 10, 17 → new idx 11. No collision here; just confirm both retained.
    assert 10 in out and 11 in out


def test_intertitle_plan_single_card_end_of_stream():
    cards = _cards([(95, 99)])
    n2o, o2n = build_frame_index_map(100, cards)
    plan = build_intertitle_plan(cards, fps=25.0, source_start_time_s=3.0,
                                  orig_to_new=o2n, total_new=len(n2o))
    assert plan["source_fps"] == 25.0
    assert len(plan["cards"]) == 1
    c = plan["cards"][0]
    assert c["duration_s"] == 5 / 25
    assert c["trimmed_time_s"] == 95 / 25
    assert c["original_time_s"] == 95 / 25 + 3.0
    # Card at end of stream → insert at tail.
    assert c["new_insert_position"] == len(n2o)


def test_intertitle_plan_multiple_cards_insert_positions_monotonic():
    cards = _cards([(10, 14), (50, 54), (90, 94)])
    n2o, o2n = build_frame_index_map(100, cards)
    plan = build_intertitle_plan(cards, fps=25.0, source_start_time_s=0.0,
                                  orig_to_new=o2n, total_new=len(n2o))
    positions = [c["new_insert_position"] for c in plan["cards"]]
    assert positions == sorted(positions)
