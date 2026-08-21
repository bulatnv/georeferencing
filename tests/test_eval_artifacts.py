"""Провенанс входа экспериментов: частичный прогон не трогает канон, позы не молчат.

Тесты по ``docs/FIX_EVAL_ARTIFACT_LEAK.md``. Конкретная авария: одиночные
прогоны ``--cases Volgograd4`` молча перезаписали ``eval_out/eval.csv`` одной
строкой, и оракульные пробы manual-кейсов либо тихо теряли кейсы, либо (форма
вторая, хуже) взяли бы позы, порождённые другим ядром, — и это не было видно
нигде. Третье правило регламента: **вход эксперимента обязан нести провенанс
наравне с весами.**
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_dataset import _cases_slug, output_csv_path  # noqa: E402
from poses_provenance import (  # noqa: E402
    PROVENANCE_FIELDS,
    PosesError,
    load_poses_with_provenance,
)


# --- Ф1: частичный прогон не пишет канонический eval.csv ----------------------

def test_full_run_keeps_the_canonical_name():
    assert output_csv_path("", "eval_out", "").name == "eval.csv"


def test_partial_run_never_lands_on_eval_csv_silently():
    """Суть утечки: --cases Volgograd4 затирал eval.csv одной строкой."""
    path = output_csv_path("", "eval_out", "Volgograd4")
    assert path.name != "eval.csv"
    assert path.name == "eval_cases_Volgograd4.csv"


def test_explicit_out_is_honored():
    assert output_csv_path("x/y.csv", "eval_out", "A,B") == Path("x/y.csv")


def test_cases_slug_is_deterministic_and_filename_safe():
    a = _cases_slug("DRZ_00755, Ufa3")
    assert a == _cases_slug("DRZ_00755, Ufa3") == "DRZ_00755-Ufa3"
    assert not set(a) & set('\\/:*?"<>| ')


def test_long_case_list_collapses_but_stays_unique():
    many = ",".join(f"case_{i:03d}" for i in range(30))
    slug = _cases_slug(many)
    assert len(slug) <= 60 and slug.startswith("case_000_and_29_more_")
    other = _cases_slug(many.replace("case_029", "case_030"))
    assert slug != other                     # хэш различает разные списки


# --- Ф2/Ф3: загрузчик поз — провенанс и отказ вместо тишины -------------------

def _write_poses(tmp_path, cases, matcher="minima_roma", sidecar=True) -> Path:
    path = tmp_path / "eval.csv"
    lines = ["case,found_lat,found_lon,heading_deg"]
    lines += [f"{c},48.7,44.5,10.0" for c in cases]
    path.write_text("\n".join(lines), encoding="utf-8")
    if sidecar:
        path.with_suffix(".config.json").write_text(
            json.dumps({"matcher": matcher}), encoding="utf-8")
    return path


def test_missing_file_raises_with_a_hint_when_required(tmp_path):
    with pytest.raises(PosesError, match="eval_dataset"):
        load_poses_with_provenance(tmp_path / "нет.csv", required={"Volgograd3"})


def test_missing_file_is_fine_when_nothing_is_required(tmp_path):
    """Кейсы с EXIF-курсом в файле поз не нуждаются — отказ был бы ложным."""
    poses, prov = load_poses_with_provenance(tmp_path / "нет.csv", required=set())
    assert poses == {} and prov["poses_matcher"] == "unknown"


def test_truncated_file_raises_without_the_flag(tmp_path):
    """Форма первая: файл затёрт частичным прогоном — molча пропускать нельзя."""
    path = _write_poses(tmp_path, ["Volgograd4"])
    with pytest.raises(PosesError, match="Volgograd3"):
        load_poses_with_provenance(path, required={"Volgograd3"})


def test_truncated_file_passes_with_the_escape_hatch(tmp_path):
    path = _write_poses(tmp_path, ["Volgograd4"])
    poses, prov = load_poses_with_provenance(
        path, required={"Volgograd3"}, allow_partial=True)
    assert "Volgograd3" not in poses and prov["poses_n_cases"] == 1


def test_provenance_reaches_the_columns(tmp_path):
    """Форма вторая: позы другого ядра легитимны, но обязаны быть видны."""
    path = _write_poses(tmp_path, ["Volgograd3"], matcher="romav2")
    poses, prov = load_poses_with_provenance(path, required={"Volgograd3"})
    assert poses["Volgograd3"][0] == pytest.approx(48.7)
    assert prov["poses_matcher"] == "romav2"
    assert prov["poses_src"].endswith("eval.csv")
    assert prov["poses_n_cases"] == 1
    assert prov["poses_mtime"] != "unknown"
    assert set(PROVENANCE_FIELDS) == set(prov)


def test_missing_sidecar_marks_matcher_unknown(tmp_path):
    path = _write_poses(tmp_path, ["Volgograd3"], sidecar=False)
    _, prov = load_poses_with_provenance(path, required={"Volgograd3"})
    assert prov["poses_matcher"] == "unknown"
