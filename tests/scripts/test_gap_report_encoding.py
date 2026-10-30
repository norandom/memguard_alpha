"""Gap reports must not depend on the machine's default text encoding."""

from pathlib import Path

from scripts.analyze_is_oos_gap import load_eval_metadata, write_csv, write_md


def test_gap_report_uses_utf8_with_a_legacy_windows_default(tmp_path, monkeypatch):
    original_open = Path.open

    def legacy_default_open(path, mode="r", buffering=-1, encoding=None, **kwargs):
        if "b" not in mode and encoding is None:
            encoding = "cp1252"
        return original_open(path, mode, buffering, encoding=encoding, **kwargs)

    monkeypatch.setattr(Path, "open", legacy_default_open)
    markdown_path = tmp_path / "gap.md"
    csv_path = tmp_path / "gap.csv"
    write_md(markdown_path, [], tmp_path)
    write_csv(csv_path, [{"model": "模型", "gap_point": 0.1}])
    assert "IS_accuracy − OOS_accuracy" in markdown_path.read_text(encoding="utf-8")
    assert "模型" in csv_path.read_text(encoding="utf-8")

    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text('{"prompt":"模型", "metadata":{"ticker":"日本"}}\n', encoding="utf-8")
    assert next(iter(load_eval_metadata(eval_path).values())) == {"ticker": "日本"}
