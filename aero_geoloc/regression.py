"""Регрессия оценки: заморозить поведение набора и ловить его изменение.

Зачем модуль ([EVAL_PLAN.md](../docs/EVAL_PLAN.md), этап F; порядок работ —
[ROADMAP.md](../docs/ROADMAP.md), фаза 0). Дальше меняется **ядро матчинга** —
самый нагруженный компонент, — а за одну сессию тихая регрессия дважды прошла
почти до конца:

* авто-перекрытие сетки потеряло ``Volgograd3`` (0.8 м → отказ) — заметили
  глазами по таблице;
* ключ кэша карты не включал перекрытие, прогон молча взял чужую карту и
  «доказал», что изменение не помогло.

Оба раза последней линией обороны был внимательный просмотр. При подмене матчера
этого не хватит: меняется сразу всё — число инлайеров, NCC, время, — и на фоне
«всё поехало» одна пропавшая локализация не бросается в глаза.

Что здесь есть
--------------
**Класс исхода** (:func:`outcome_of`) — огрубление строки прогона до того, что
нас реально волнует. Ошибка в метрах шумит от прогона к прогону, а вот «место
принято и оно верное» против «место принято и оно неверное» — не шумит.

**Порядок исходов** (:data:`OUTCOME_RANK`) — какой исход лучше какого. Регрессия
определяется как **падение по этому порядку**, а не как «числа изменились».

**Ложное срабатывание рушит сборку всегда** — независимо от того, что улучшилось
в остальном. Это прямое следствие инварианта «честный отказ дороже красивой
точки» (`docs/ARCHITECTURE.md`): размен «+2 верных ценой одного ложного» не
является улучшением, и порядок исходов не должен давать его совершить.

**Конфигурация прогона — часть золота.** Сравнивать прогон с радиусом 1.5 км
против золота с радиусом 2 км бессмысленно, но по одной таблице это не видно.
Поэтому конфигурация замораживается вместе с исходами, и расхождение по ключам,
меняющим смысл сравнения, объявляется явно.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "OUTCOME_RANK", "OUTCOME_RU", "CONFIG_KEYS",
    "CaseExpectation", "Golden", "CaseVerdict", "RegressionReport",
    "outcome_of", "freeze", "compare", "load_golden", "save_golden",
]

#: Исходы кейса по возрастанию «хорошести». Регрессия = падение по этой шкале.
#:
#: Обоснование нескольких неочевидных мест:
#:
#: * ``accepted_wrong`` (ложное срабатывание) стоит **ниже отказа и ниже
#:   аварии**: уверенно-неверная точка вреднее и молчания, и падения скрипта —
#:   на неё полагаются. Отдельно от порядка оно ещё и рушит сборку безусловно.
#: * ``pose_wrong_gated`` (поза найдена, место неверное, гейт её отверг) стоит
#:   вровень с отказом: снаружи это одно и то же — координат не выдали.
#: * ``*_unverified`` (истины нет) выше отказа, но ниже подтверждённо верного:
#:   такой исход сам по себе не доказательство, его надо смотреть глазами.
OUTCOME_RANK: dict[str, int] = {
    "accepted_wrong": -20,      # ЛОЖНОЕ: принято, но не то место
    "error": -10,               # кейс упал с исключением
    "refused_floor1": 0,        # отказ, верная клетка не доехала до Этажа 2
    "refused_floor2": 0,        # отказ, клетка была, поза не сошлась
    "refused": 0,               # отказ, виновник не определён
    "pose_wrong_gated": 0,      # поза неверная, гейт отверг — корректный отказ
    "pose_gated_unverified": 1,
    "pose_correct_gated": 1,    # ВЕРНАЯ поза, но гейт не пропустил
    "accepted_unverified": 2,
    "accepted_correct": 3,      # принято и место верное
}

OUTCOME_RU: dict[str, str] = {
    "accepted_wrong": "ЛОЖНОЕ (принято не то место)",
    "error": "авария кейса",
    "refused_floor1": "отказ (Этаж 1)",
    "refused_floor2": "отказ (Этаж 2)",
    "refused": "отказ",
    "pose_wrong_gated": "гейт отверг неверную",
    "pose_gated_unverified": "поза есть, гейт отверг, истины нет",
    "pose_correct_gated": "ГЕЙТ отверг ВЕРНУЮ",
    "accepted_unverified": "принято, истины нет",
    "accepted_correct": "принято, место верное",
}

#: Ключи конфигурации, меняющие смысл сравнения. Прогон с другим радиусом или
#: другим матчером — это не регрессия, а другой эксперимент, и путать их нельзя.
CONFIG_KEYS = (
    "manifest", "matcher", "matcher_max_side", "photometric", "min_photometric",
    "radius_km", "cell_px", "overlap", "pca_dim", "max_fine_window_px",
    "top_k", "min_inliers", "rotation_step", "correct_m", "manual_tol_frac",
    "offset_km", "sigma_m",
)


def outcome_of(row: dict) -> str:
    """Класс исхода строки прогона (см. ``FIELDS`` в ``scripts/eval_dataset.py``).

    Огрубление намеренное: ошибка в метрах и число инлайеров плавают от прогона
    к прогону (RANSAC недетерминирован), а класс исхода — нет. Сравнивать надо
    то, что устойчиво.
    """
    status = str(row.get("status", "")).strip()
    if status in ("ошибка", "error"):
        return "error"

    if status not in ("localized", "low_confidence"):
        blame = str(row.get("blame", ""))
        if blame.startswith("Этаж 1"):
            return "refused_floor1"
        if blame.startswith("Этаж 2"):
            return "refused_floor2"
        return "refused"

    accepted = str(row.get("accepted", "")).strip() in ("1", "True", "true")
    correct = str(row.get("correct", "")).strip()
    if correct == "":
        return "accepted_unverified" if accepted else "pose_gated_unverified"
    is_correct = correct in ("1", "True", "true")
    if accepted:
        return "accepted_correct" if is_correct else "accepted_wrong"
    return "pose_correct_gated" if is_correct else "pose_wrong_gated"


def _num(value) -> float | None:
    """Число из ячейки CSV; пустая ячейка — это ``None``, а не ноль."""
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class CaseExpectation:
    """Чего мы ждём от кейса. Числа — справочные, вердикт даёт ``outcome``."""

    name: str
    outcome: str
    error_m: float | None = None
    true_cell_rank: int | None = None
    n_inliers: int | None = None
    ncc: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        out: dict = {"outcome": self.outcome}
        for key in ("error_m", "true_cell_rank", "n_inliers", "ncc"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.note:
            out["note"] = self.note
        return out


@dataclass(frozen=True)
class Golden:
    """Замороженное поведение набора: конфигурация прогона + исход каждого кейса."""

    config: dict = field(default_factory=dict)
    cases: dict[str, CaseExpectation] = field(default_factory=dict)
    note: str = ""

    @property
    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for exp in self.cases.values():
            counts[exp.outcome] = counts.get(exp.outcome, 0) + 1
        return counts


@dataclass(frozen=True)
class CaseVerdict:
    """Что стало с одним кейсом. ``severity`` — единственное, что решает судьбу сборки."""

    name: str
    was: str | None            # исход в золоте (None — кейса там не было)
    now: str | None            # исход в прогоне (None — кейс не прогонялся)
    severity: str              # "регрессия" | "ухудшение" | "улучшение" | "ok" | "новый" | "пропал"
    detail: str = ""

    @property
    def is_failure(self) -> bool:
        return self.severity == "регрессия"


@dataclass
class RegressionReport:
    verdicts: list[CaseVerdict] = field(default_factory=list)
    config_diff: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[CaseVerdict]:
        return [v for v in self.verdicts if v.is_failure]

    @property
    def warnings(self) -> list[CaseVerdict]:
        return [v for v in self.verdicts if v.severity in ("ухудшение", "новый", "пропал")]

    @property
    def improvements(self) -> list[CaseVerdict]:
        return [v for v in self.verdicts if v.severity == "улучшение"]

    @property
    def passed(self) -> bool:
        """Сборка проходит, только если нет регрессий И совпала конфигурация.

        Расхождение конфигурации не «предупреждение»: сравнение с золотом,
        снятым при других параметрах, ничего не доказывает, а выглядит как
        доказательство. Такой прогон честнее объявить непроведённым.
        """
        return not self.failures and not self.config_diff


def freeze(rows: list[dict], config: dict, *, note: str = "") -> Golden:
    """Снять золото с прогона: исход и справочные числа по каждому кейсу."""
    cases: dict[str, CaseExpectation] = {}
    for row in rows:
        name = str(row.get("case", "")).strip()
        if not name:
            continue
        rank = _num(row.get("true_cell_rank"))
        inliers = _num(row.get("n_inliers"))
        error = _num(row.get("error_m"))
        ncc = _num(row.get("photometric"))
        cases[name] = CaseExpectation(
            name=name,
            outcome=outcome_of(row),
            error_m=round(error, 1) if error is not None else None,
            true_cell_rank=int(rank) if rank is not None else None,
            n_inliers=int(inliers) if inliers is not None else None,
            ncc=round(ncc, 4) if ncc is not None else None,
            note=str(row.get("blame", "")),
        )
    return Golden(config=dict(config), cases=cases, note=note)


def _config_diff(golden: Golden, config: dict) -> list[str]:
    diff = []
    for key in CONFIG_KEYS:
        if key not in golden.config and key not in config:
            continue
        was, now = golden.config.get(key), config.get(key)
        if was is None and now is None:
            continue
        if str(was) != str(now):
            diff.append(f"{key}: золото {was!r} ≠ прогон {now!r}")
    return diff


def _error_verdict(exp: CaseExpectation, row: dict, slack_m: float, slack_frac: float) -> str:
    """Насколько уехала ошибка при неизменном исходе.

    Полоса допуска нужна, потому что RANSAC недетерминирован и метры гуляют сами
    по себе. Но «класс тот же» ещё не значит «всё в порядке»: рост ошибки с 0.8 м
    до 45 м формально остаётся верной локализацией, а фактически означает, что
    поза поехала. Такое помечается **ухудшением** — сборку не рушит, но в глаза
    бросается.
    """
    now = _num(row.get("error_m"))
    if exp.error_m is None or now is None:
        return ""
    # У позы, которую гейт правильно отверг как неверную, «ошибка» — это расстояние
    # до истины у заведомо чужого места. Насколько именно оно чужое, нам не важно:
    # сравнение таких чисел между прогонами даёт шум, а не сигнал.
    if exp.outcome in ("pose_wrong_gated", "accepted_wrong"):
        return ""
    allowed = max(exp.error_m + slack_m, exp.error_m * (1.0 + slack_frac))
    if now > allowed:
        return f"ошибка {exp.error_m} → {now:.1f} м (допуск {allowed:.1f})"
    if now + slack_m < exp.error_m and now * (1.0 + slack_frac) < exp.error_m:
        return f"ошибка {exp.error_m} → {now:.1f} м — лучше"
    return ""


def compare(rows: list[dict], golden: Golden, config: dict, *,
            slack_m: float = 10.0, slack_frac: float = 0.5) -> RegressionReport:
    """Сравнить прогон с золотом. Вердикт — по классу исхода, не по числам."""
    report = RegressionReport(config_diff=_config_diff(golden, config))
    by_name = {str(r.get("case", "")).strip(): r for r in rows}

    for name in sorted(set(golden.cases) | set(by_name)):
        exp = golden.cases.get(name)
        row = by_name.get(name)

        if exp is None:
            report.verdicts.append(CaseVerdict(
                name, None, outcome_of(row), "новый",
                "кейса нет в золоте — заморозить командой --freeze"))
            continue
        if row is None:
            report.verdicts.append(CaseVerdict(
                name, exp.outcome, None, "пропал",
                "кейс есть в золоте, но не прогонялся"))
            continue

        now = outcome_of(row)
        was_rank = OUTCOME_RANK.get(exp.outcome, 0)
        now_rank = OUTCOME_RANK.get(now, 0)

        if now == "accepted_wrong":
            # Ложное принято — рушим сборку даже если в золоте было то же самое:
            # золото с ложным срабатыванием не должно тихо жить дальше.
            error = _num(row.get("error_m"))
            report.verdicts.append(CaseVerdict(
                name, exp.outcome, now, "регрессия",
                f"ЛОЖНОЕ СРАБАТЫВАНИЕ, ошибка {error:.0f} м" if error is not None
                else "ЛОЖНОЕ СРАБАТЫВАНИЕ"))
            continue
        if now_rank < was_rank:
            report.verdicts.append(CaseVerdict(
                name, exp.outcome, now, "регрессия",
                f"{OUTCOME_RU.get(exp.outcome, exp.outcome)} → {OUTCOME_RU.get(now, now)}"))
            continue
        if now_rank > was_rank:
            report.verdicts.append(CaseVerdict(
                name, exp.outcome, now, "улучшение",
                f"{OUTCOME_RU.get(exp.outcome, exp.outcome)} → {OUTCOME_RU.get(now, now)}"))
            continue

        detail = _error_verdict(exp, row, slack_m, slack_frac)
        if detail.endswith("— лучше"):
            report.verdicts.append(CaseVerdict(name, exp.outcome, now, "улучшение", detail))
        elif detail:
            report.verdicts.append(CaseVerdict(name, exp.outcome, now, "ухудшение", detail))
        else:
            report.verdicts.append(CaseVerdict(name, exp.outcome, now, "ok"))
    return report


def load_golden(path) -> Golden:
    """Прочитать золото из YAML."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cases = {}
    for name, spec in (data.get("cases") or {}).items():
        spec = spec or {}
        cases[str(name)] = CaseExpectation(
            name=str(name),
            outcome=str(spec.get("outcome", "refused")),
            error_m=spec.get("error_m"),
            true_cell_rank=spec.get("true_cell_rank"),
            n_inliers=spec.get("n_inliers"),
            ncc=spec.get("ncc"),
            note=str(spec.get("note", "")),
        )
    return Golden(config=dict(data.get("config") or {}), cases=cases,
                  note=str(data.get("note", "")))


def save_golden(golden: Golden, path, *, header: str = "") -> None:
    """Записать золото в YAML — файл читается глазами, поэтому не JSON."""
    import yaml

    payload = {
        "note": golden.note,
        "config": dict(golden.config),
        "cases": {name: exp.to_dict() for name, exp in sorted(golden.cases.items())},
    }
    with open(path, "w", encoding="utf-8") as fh:
        if header:
            fh.write(header if header.endswith("\n") else header + "\n")
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False, width=100)
