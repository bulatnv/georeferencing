"""V-тесты привязки к производительным ядрам."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpu_affinity import cores_by_class, current_mask, performance_mask  # noqa: E402

windows_only = pytest.mark.skipif(not sys.platform.startswith("win"),
                                  reason="привязка ядер — про Windows")


@windows_only
def test_classes_do_not_share_logical_cpus():
    """Логический процессор принадлежит ровно одному классу ядер."""
    by = cores_by_class()
    seen: set[int] = set()
    for cpus in by.values():
        assert not (seen & set(cpus))
        seen |= set(cpus)
    assert seen, "система не сообщила ни одного ядра"


@windows_only
def test_performance_mask_holds_only_top_class():
    """В маске только процессоры старшего класса — и все они.

    Проверка аналитическая: маска обязана быть побитовым образом списка
    `cores_by_class()[max]`, а не «первыми N битами» — на другой машине
    нумерация может быть иной.
    """
    by = cores_by_class()
    mask = performance_mask()
    if len(by) < 2:                       # однородный CPU: делить нечего
        assert mask == 0
        return
    top = max(by)
    expect = 0
    for cpu in by[top]:
        expect |= 1 << cpu
    assert mask == expect
    for eff, cpus in by.items():
        if eff == top:
            continue
        for cpu in cpus:                  # медленные ядра в маску не попали
            assert not mask >> cpu & 1


@windows_only
def test_mask_is_subset_of_system_mask():
    """Маска производительных ядер не выходит за пределы доступных процессу."""
    mask = performance_mask()
    if mask:
        assert mask & current_mask() == mask or current_mask() == mask
