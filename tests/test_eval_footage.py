import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import eval_footage as ef  # noqa: E402


def test_parse_seconds():
    assert ef.parse_seconds("83.5") == 83.5
    assert ef.parse_seconds("1:23.5") == 83.5
    assert ef.parse_seconds("1:02:03") == 3723
    with pytest.raises(ValueError):
        ef.parse_seconds("abc")


def test_load_truth(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("﻿Video, Seconds_Start,seconds_end,Plate,condition\n"
                 "a.mp4,1,0:04,abc 123,night\n\nb.mp4,2,3,XYZ789,\n", encoding="utf-8")
    rows = ef.load_truth(p)
    assert rows == [ef.Truth("a.mp4", 1, 4, "ABC123", "night"), ef.Truth("b.mp4", 2, 3, "XYZ789", "unspecified")]


def test_load_truth_errors(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("video,plate\na.mp4,ABC123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        ef.load_truth(p)
    p.write_text("video,seconds_start,seconds_end,plate\na.mp4,5,2,ABC123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        ef.load_truth(p)


def test_edit_distance():
    assert ef.edit_distance("ABC123", "ABC123") == 0
    assert ef.edit_distance("ABC123", "ABC124") == 1
    assert ef.edit_distance("ABC123", "AB123") == 1
    assert ef.edit_distance("", "ABC") == 3


def test_matching_outcomes():
    truth, reports = ef.sample_data()
    m = ef.match_reports(truth, reports, slack=1.5)
    assert [o.status for o in m.outcomes] == [ef.EXACT, ef.WRONG, ef.MISSED, ef.MISSED]
    wrong = m.outcomes[1]
    assert (wrong.got, wrong.distance) == ("XYZ780", 1)
    assert m.outcomes[2].distance == 7  # unreadable: whole plate lost
    assert [r.plate for r in m.extras] == ["QQQ111"]


def test_time_window_and_slack():
    t = [ef.Truth("a.mp4", 10, 12, "AAA111")]
    inside = ef.match_reports(t, [ef.Report("a.mp4", 13.0, "AAA111")], slack=1.5)
    assert inside.outcomes[0].status == ef.EXACT
    outside = ef.match_reports(t, [ef.Report("a.mp4", 14.0, "AAA111")], slack=1.5)
    assert outside.outcomes[0].status == ef.MISSED and len(outside.extras) == 1
    other_video = ef.match_reports(t, [ef.Report("b.mp4", 11, "AAA111")])
    assert other_video.outcomes[0].status == ef.MISSED


def test_best_candidate_and_repeats():
    t = [ef.Truth("a.mp4", 10, 14, "AAA111")]
    reps = [ef.Report("a.mp4", 10.5, "AAA117"), ef.Report("a.mp4", 12, "AAA111"),
            ef.Report("a.mp4", 13, "AAA111"), ef.Report("a.mp4", 13.5, "ZZZ999")]
    m = ef.match_reports(t, reps)
    assert m.outcomes[0].status == ef.EXACT
    # Repeat of the right plate is ignored; the stray wrong one and the near-miss are extras.
    assert sorted(r.plate for r in m.extras) == ["AAA117", "ZZZ999"]


def test_report_used_once():
    t = [ef.Truth("a.mp4", 10, 12, "AAA111"), ef.Truth("a.mp4", 12.5, 14, "BBB222")]
    m = ef.match_reports(t, [ef.Report("a.mp4", 12.2, "AAA111")])
    assert [o.status for o in m.outcomes] == [ef.EXACT, ef.MISSED]


def test_stats_per_condition():
    truth, reports = ef.sample_data()
    stats = ef.compute_stats(ef.match_reports(truth, reports))
    allv = stats["ALL"]
    assert (allv.vehicles, allv.exact, allv.wrong, allv.missed, allv.extras) == (4, 1, 1, 2, 1)
    assert allv.edits == 0 + 1 + 7 + 7 and allv.chars == 27
    assert allv.cer == pytest.approx(15 / 27)
    assert stats["day"].vehicles == 2 and stats["day"].exact == 1 and stats["day"].missed == 1
    assert stats["night"].wrong == 1 and stats["rain"].missed == 1


def test_source_seconds():
    assert ef.source_seconds("video gate.mp4 at 1:23") == 83
    assert ef.source_seconds("video gate.mp4 at 1:02:03") == 3723
    assert ef.source_seconds(None) is None


def test_outputs(tmp_path):
    truth, reports = ef.sample_data()
    m = ef.match_reports(truth, reports)
    md, mism = ef.write_outputs(tmp_path / "out", m, 1.5)
    text = md.read_text(encoding="utf-8")
    assert "| **ALL** | 4 | 1 (25%)" in text and "XYZ789 | XYZ780 | 1" in text
    rows = list(csv.DictReader(open(mism, encoding="utf-8")))
    assert [r["kind"] for r in rows] == ["wrong", "missed", "missed", "extra"]
    assert rows[0]["expected"] == "XYZ789" and rows[0]["got"] == "XYZ780"


def test_selftest_runs(tmp_path, capsys):
    assert ef.main(["--selftest", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "report.md").exists()
