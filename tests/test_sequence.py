"""Тесты последовательностного режима (фаза 5): VO, EKF, привязка дрейфа якорями.

VO проверяется против **известного** движения (формула, не «как получилось»);
EKF — что редкие уверенные фиксы ограничивают дрейф, а бедные кадры между ними
не заставляют угадывать.
"""

from __future__ import annotations

import numpy as np
import pytest

from aero_geoloc.sequence import (
    AbsoluteFix,
    EKFState,
    TrajectoryEKF,
    VOStep,
    estimate_vo,
    localize_sequence,
)
from aero_geoloc.testbench import (
    SampleSpec,
    default_camera,
    generate_sample,
    generate_trajectory,
    make_synthetic_scene,
)
from aero_geoloc.localize import normalize_gray


@pytest.fixture(scope="module")
def scene():
    return make_synthetic_scene(2048, seed=0)


@pytest.fixture(scope="module")
def camera():
    return default_camera(512)


# --- визуальная одометрия ---------------------------------------------------


def test_estimate_vo_recovers_known_motion(scene, camera):
    """VO восстанавливает заданное смещение и поворот между кадрами."""
    mpp = scene.georef.mpp
    dx, dy, yaw_a, yaw_b = 40.0, -20.0, 10.0, 25.0
    prev = normalize_gray(generate_sample(scene, camera, SampleSpec(yaw_deg=yaw_a), reference_size=1000).query)
    curr = normalize_gray(
        generate_sample(scene, camera, SampleSpec(yaw_deg=yaw_b, center_offset_px=(dx, dy)), reference_size=1000).query
    )
    vo = estimate_vo(prev, curr, camera, 600.0, yaw_a)
    assert vo is not None
    assert vo.delta_yaw_deg == pytest.approx(yaw_b - yaw_a, abs=0.3)
    assert vo.delta_east_m == pytest.approx(dx * mpp, abs=0.3)
    assert vo.delta_north_m == pytest.approx(-dy * mpp, abs=0.3)


def test_estimate_vo_returns_none_on_incompatible_frames(scene, camera):
    prev = normalize_gray(generate_sample(scene, camera, SampleSpec(), reference_size=1000).query)
    blank = np.full((512, 512), 127, np.uint8)
    assert estimate_vo(prev, blank, camera, 600.0, 0.0) is None


# --- EKF --------------------------------------------------------------------


def test_ekf_update_pulls_state_toward_fix():
    ekf = TrajectoryEKF(EKFState(0.0, 0.0, 0.0, np.diag([100.0, 100.0, 25.0])))
    ekf.update(AbsoluteFix(10.0, -5.0, 8.0, position_sigma_m=1.0, heading_sigma_deg=1.0))
    # Фикс точнее приора → состояние подтягивается к нему.
    assert ekf.state.east_m == pytest.approx(10.0, abs=0.5)
    assert ekf.state.north_m == pytest.approx(-5.0, abs=0.5)
    assert ekf.state.heading_deg == pytest.approx(8.0, abs=0.5)
    assert ekf.state.position_sigma_m < 10.0  # неопределённость упала


def test_predict_missing_grows_covariance_without_moving():
    ekf = TrajectoryEKF(EKFState(3.0, 4.0, 30.0, np.eye(3)))
    before = ekf.state.position_sigma_m
    ekf.predict_missing(position_drift_sigma_m=5.0)
    assert (ekf.state.east_m, ekf.state.north_m) == (3.0, 4.0)  # не двигаемся
    assert ekf.state.position_sigma_m > before  # но растём в неопределённости


def test_ekf_heading_wraps_around_north():
    ekf = TrajectoryEKF(EKFState(0.0, 0.0, 350.0, np.diag([1.0, 1.0, 100.0])))
    ekf.update(AbsoluteFix(0.0, 0.0, 10.0, position_sigma_m=1.0, heading_sigma_deg=1.0))
    # 350° и 10° различаются на 20°, а не на 340° — курс должен уйти к ~5..10°.
    assert -10.0 < ekf.state.heading_deg < 30.0


# --- слияние: якоря ограничивают дрейф (детерминированно) --------------------


def test_ekf_anchors_bound_biased_drift():
    """Смещённый VO уводит чистый дедрекон; редкие якоря удерживают траекторию.

    Свойство слияния тестируем на инжектированных VO-шагах (детерминированно):
    истинное движение +1 м/шаг на восток, но VO смещён на +0.3 м/шаг. Чистый
    EKF копит смещение (≈ N·0.3 м), а якоря каждые 4 шага сбрасывают его.
    """
    n = 20
    bias = 0.3
    pure = TrajectoryEKF(EKFState(0.0, 0.0, 0.0, np.diag([1.0, 1.0, 1.0])))
    anchored = TrajectoryEKF(EKFState(0.0, 0.0, 0.0, np.diag([1.0, 1.0, 1.0])))

    for i in range(1, n + 1):
        step = VOStep(1.0 + bias, 0.0, 0.0, scale=1.0, n_inliers=50, position_sigma_m=0.5)
        pure.predict(step)
        anchored.predict(step)
        if i % 4 == 0:
            anchored.update(AbsoluteFix(float(i), 0.0, 0.0, position_sigma_m=0.5, heading_sigma_deg=1.0))

    true_east = float(n)
    assert abs(pure.state.east_m - true_east) == pytest.approx(n * bias, rel=0.2)  # копит смещение
    assert abs(anchored.state.east_m - true_east) < 1.0  # якоря удержали
    assert anchored.state.position_sigma_m < pure.state.position_sigma_m


def test_sequence_end_to_end_tracks_trajectory(scene, camera):
    """Сквозной VO+EKF на реальных кадрах: траектория отслеживается точно."""
    mpp = scene.georef.mpp
    waypoints = [(-330.0 + 60.0 * i, 40.0 * np.sin(i / 2.0), 5.0 * np.sin(i / 3.0)) for i in range(12)]
    frames = [normalize_gray(s.query) for s in generate_trajectory(scene, camera, waypoints, reference_size=1000)]
    truth = [(ox * mpp, -oy * mpp, yaw) for ox, oy, yaw in waypoints]
    init = EKFState(truth[0][0], truth[0][1], truth[0][2], np.diag([1.0, 1.0, 1.0]))

    states = localize_sequence(frames, camera, init, altitude_m=600.0)
    max_err = max(np.hypot(s.east_m - te, s.north_m - tn) for s, (te, tn, _) in zip(states, truth))
    assert max_err < 2.0  # VO держит траекторию в пределах метров без якорей


def test_sequence_survives_a_poor_frame(scene, camera):
    mpp = scene.georef.mpp
    waypoints = [(-330.0 + 60.0 * i, 40.0 * np.sin(i / 2.0), 5.0 * np.sin(i / 3.0)) for i in range(12)]
    frames = [normalize_gray(s.query) for s in generate_trajectory(scene, camera, waypoints, reference_size=1000)]
    truth = [(ox * mpp, -oy * mpp, yaw) for ox, oy, yaw in waypoints]
    frames[5] = np.full((512, 512), 127, np.uint8)  # один бедный кадр — VO там провалится
    init = EKFState(truth[0][0], truth[0][1], truth[0][2], np.diag([1.0, 1.0, 1.0]))

    states = localize_sequence(frames, camera, init, altitude_m=600.0)
    assert len(states) == len(frames)  # не падаем
    # После бедного кадра неопределённость обязана вырасти (честно, а не угадывать).
    assert states[6].position_sigma_m > states[4].position_sigma_m
