from hackathon.egoflow.hand_interaction import detect_aborted_reaches


def _rows(points):
    return [
        {"timestamp_sec": index / 8, "hand": "right", "x": x, "y": y, "aperture": 1.0}
        for index, (x, y) in enumerate(points)
    ]


def test_aborted_reach_detects_moving_redirection():
    points = [(0.10 + index * 0.025, 0.4) for index in range(9)]
    points += [(0.30 - index * 0.018, 0.4 + index * 0.022) for index in range(1, 10)]
    events = detect_aborted_reaches(_rows(points), sample_fps=8, max_candidates=2)
    assert events
    assert events[0]["label"] == "aborted_reach"
    assert events[0]["direction_change_deg"] > 75


def test_stationary_waiting_and_straight_motion_do_not_trigger():
    stationary = _rows([(0.4, 0.4)] * 20)
    straight = _rows([(0.1 + index * 0.02, 0.4) for index in range(20)])
    assert detect_aborted_reaches(stationary, sample_fps=8) == []
    assert detect_aborted_reaches(straight, sample_fps=8) == []
