"""Тесты последовательностного режима (фаза 5): VO, EKF, привязка дрейфа якорями.

VO проверяется против **известного** движения (формула, не «как получилось»);
EKF — что редкие уверенные фиксы ограничивают дрейф, а бедные кадры между ними
не заставляют угадывать.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

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


def test_estimate_vo_prerotate_preserves_recovered_motion(scene, camera):
    """Предповорот кадра (для разворотов) не искажает восстановленное движение.

    На развороте соседние кадры повёрнуты друг относительно друга; предповорот
    на разницу курсов выравнивает их для матчера, а точки возвращаются в исходные
    координаты — так что VO обязана дать то же смещение/поворот, что и без него.
    """
    mpp = scene.georef.mpp
    dx, dy, yaw_a, yaw_b = 35.0, -15.0, 20.0, 90.0  # крупный поворот, как на развороте
    prev = normalize_gray(generate_sample(scene, camera, SampleSpec(yaw_deg=yaw_a), reference_size=1100).query)
    curr = normalize_gray(
        generate_sample(scene, camera, SampleSpec(yaw_deg=yaw_b, center_offset_px=(dx, dy)), reference_size=1100).query
    )
    vo = estimate_vo(prev, curr, camera, 600.0, yaw_a, prerotate_deg=yaw_a - yaw_b)
    assert vo is not None
    assert vo.delta_yaw_deg == pytest.approx(yaw_b - yaw_a, abs=0.5)
    assert vo.delta_east_m == pytest.approx(dx * mpp, abs=0.5)
    assert vo.delta_north_m == pytest.approx(-dy * mpp, abs=0.5)


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


# --- VO на реальной бортовой серии (gated) ----------------------------------

_UFA = Path(__file__).resolve().parents[1] / "for_binding" / "ufa"
_HAS_LG = importlib.util.find_spec("lightglue") is not None
_HAS_PIL = importlib.util.find_spec("PIL") is not None


@pytest.mark.skipif(
    not (_UFA.exists() and _HAS_LG and _HAS_PIL),
    reason="нужны серия for_binding/ufa + LightGlue",
)
def test_vo_tracks_real_ufa_through_turn():
    """VO кадр-к-кадру на реальной серии трекает GPS через разворот, с малым дрейфом.

    Соседние кадры дрона почти без appearance gap (та же камера/полёт), поэтому VO
    работает. Курс приводится к истинному северу (склонение Уфы ~+14.5°), а на
    развороте соседние кадры выравниваются предповоротом на разницу курсов
    (``headings``) — без этого не-инвариантный матчер провалился бы на повороте.
    """
    import cv2

    from aero_geoloc.camera import Camera
    from aero_geoloc.drone import load_drone_shot
    from aero_geoloc.geo import haversine_m
    from aero_geoloc.matcher import LightGlueMatcher

    files = sorted(_UFA.glob("*.JPG"))[:30]  # захватывает разворот между полосами
    shots = [load_drone_shot(f, magnetic_declination_deg=14.5) for f in files]
    shots = [s for s in shots if s.is_nadir]
    if len(shots) < 20:
        pytest.skip("мало надирных кадров в серии")
    assert max(abs((shots[i].yaw_deg - shots[i - 1].yaw_deg + 180) % 360 - 180)
               for i in range(1, len(shots))) > 45.0  # в серии есть разворот

    lat0, lon0 = shots[0].true_lat, shots[0].true_lon
    truth = [
        (haversine_m(lat0, lon0, lat0, s.true_lon) * np.sign(s.true_lon - lon0),
         haversine_m(lat0, lon0, s.true_lat, lon0) * np.sign(s.true_lat - lat0))
        for s in shots
    ]
    altitude = float(np.mean([s.altitude_m for s in shots]))
    scale = 1024 / shots[0].camera.image_width
    frames = [
        normalize_gray(cv2.resize(s.image_bgr, (1024, round(s.camera.image_height * scale)),
                                  interpolation=cv2.INTER_AREA))
        for s in shots
    ]
    camera = Camera(frames[0].shape[1], frames[0].shape[0], fov_deg=shots[0].camera.fov_deg)
    init = EKFState(0.0, 0.0, shots[0].yaw_deg, np.diag([1.0, 1.0, 1.0]))

    states = localize_sequence(frames, camera, init, altitude_m=altitude,
                               matcher=LightGlueMatcher(), min_inliers=15,
                               headings=[s.yaw_deg for s in shots])
    path_len = sum(np.hypot(truth[i + 1][0] - truth[i][0], truth[i + 1][1] - truth[i][1])
                   for i in range(len(truth) - 1))
    final_drift = np.hypot(states[-1].east_m - truth[-1][0], states[-1].north_m - truth[-1][1])
    assert final_drift < 0.08 * path_len  # дрейф < 8% пути через разворот (реально ~3%)
