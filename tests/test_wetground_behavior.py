"""六个修复点的行为测试（契约测试之外的不变量验证）。"""

import math

import pytest

import gameplay.wetground_friction_sync as mod


def fresh_system(material="concrete", name="ground", wetness=0.0):
    system = mod.GroundSystem()
    surface = system.add_surface(name, material, wetness=wetness)
    return system, surface


# 1. 湿度按固定 tick 累积 ---------------------------------------------------


@pytest.mark.parametrize("fps", [30, 60, 120, 144, 240])
def test_wetness_is_independent_of_frame_rate(fps):
    ref_system, ref = fresh_system()
    ref_system.set_rain(1.0)
    ref_system.accumulate(1.0)

    system, surface = fresh_system()
    system.set_rain(1.0)
    for _ in range(fps):
        system.accumulate(1.0 / fps)

    assert system.ticks == ref_system.ticks == 30
    assert surface.wetness == pytest.approx(ref.wetness, abs=1e-12)


def test_sub_tick_time_is_carried_not_lost():
    system, surface = fresh_system()
    system.set_rain(1.0)
    for _ in range(60):
        system.accumulate(1.0 / 60.0)  # 每帧半个 tick，不能丢
    assert system.ticks == 30
    assert surface.wetness > 0.0


def test_rain_intensity_off_means_no_gain():
    system, surface = fresh_system()
    system.set_rain(0.0)
    system.accumulate(1.0)
    assert surface.wetness == 0.0


# 2. 物理与渲染同源 --------------------------------------------------------


def test_physics_and_render_read_the_same_snapshot():
    system, surface = fresh_system(wetness=0.6)
    state = surface.snapshot()
    friction = mod.apply_friction(surface, system=system)
    roughness = mod.render(surface, system=system)

    assert friction == state.physics_friction
    assert roughness == state.roughness
    assert surface.last_physics_version == surface.last_render_version == surface.version
    assert mod.validate_grip(surface, system=system)["ok"]


def test_wet_ground_actually_lowers_friction():
    system, surface = fresh_system(wetness=0.0)
    dry_friction = mod.apply_friction(surface, system=system)
    surface.wetness = 1.0
    surface.version += 1
    wet_friction = mod.apply_friction(surface, system=system)

    assert wet_friction < dry_friction  # 不再出现"视觉打滑、手感干燥"
    assert mod.validate_grip(surface, system=system)["ok"]


# 3. 衰减按材质分档 --------------------------------------------------------


def test_materials_decay_at_different_rates():
    tiers = {}
    for name in ("pool", "concrete", "grass"):
        system = mod.GroundSystem()
        surface = system.add_surface(name, name if name != "pool" else "water", wetness=1.0)
        system.dry_surface(surface, 1.0)
        tiers[name] = surface.wetness

    assert tiers["pool"] > tiers["concrete"] > tiers["grass"]
    assert not math.isclose(tiers["pool"], tiers["concrete"])


def test_decay_report_exposes_tiers():
    report = mod.decay_by_material(1.0)
    tiers = {key: item["tier"] for key, item in report["materials"].items()}
    assert tiers["water"] == "slow"
    assert tiers["grass"] == "fast"
    assert len(set(tiers.values())) >= 2


# 4. 材质切换释放贴花 ------------------------------------------------------


def test_material_switch_releases_old_decals():
    system, surface = fresh_system(wetness=0.8)
    system.accumulate(1.0 / 30.0)  # 生成湿滑贴花
    assert len(surface.live_decals()) == 1

    old_decal = surface.live_decals()[0]
    mod.set_material(surface, "grass", system=system)

    assert old_decal.released
    assert old_decal not in surface.decals  # 没有僵尸贴花
    assert surface.live_decals() == []
    assert surface.released_decals == 1


def test_switch_back_to_dry_leaves_no_stale_decal():
    system, surface = fresh_system(wetness=0.8)
    system.accumulate(1.0 / 30.0)
    mod.set_material(surface, "metal", system=system)
    mod.set_material(surface, "concrete", system=system)
    system.dry_surface(surface, 30.0)

    assert surface.wetness == 0.0
    assert surface.live_decals() == []
    assert mod.audit(system)["ok"]


def test_release_decal_is_idempotent():
    system, surface = fresh_system(wetness=0.5)
    decal = surface.add_decal()
    first = mod.release_decal(surface, decal, system=system)
    second = mod.release_decal(surface, decal, system=system)

    assert first["released"] == 1
    assert second["released"] == 0
    assert decal.release_count == 1


# 5. 音效读当帧状态 --------------------------------------------------------


def test_footstep_audio_uses_current_wetness():
    system, surface = fresh_system(wetness=0.9)
    step = mod.footstep(surface, (1.0, 0.0, 2.0), system=system)

    assert step["audio"] == "splash"
    assert step["source_version"] == surface.version
    assert step["wetness"] == surface.wetness
    assert step["footprint"] is True
    assert len(surface.footprints) == 1


def test_footstep_follows_state_change_immediately():
    system, surface = fresh_system(wetness=0.9)
    first = mod.footstep(surface, (0.0, 0.0, 0.0), system=system)
    surface.wetness = 0.0
    surface.version += 1
    second = mod.footstep(surface, (0.0, 0.0, 0.0), system=system)

    assert first["audio"] == "splash"
    assert second["audio"] == "dry"  # 不滞后一帧
    assert second["source_version"] == surface.version


# 6. 干燥后清脚印 ----------------------------------------------------------


def test_drying_clears_footprints_and_decals():
    system, surface = fresh_system(wetness=0.9)
    system.accumulate(1.0 / 30.0)
    for i in range(3):
        mod.footstep(surface, (float(i), 0.0, 0.0), system=system)
    assert len(surface.footprints) == 3
    assert surface.live_decals()

    result = mod.dry(surface, 60.0, system=system)

    assert result["wetness"] == 0.0
    assert result["footprints"] == 0
    assert result["live_decals"] == 0
    assert surface.cleared_footprints == 3
    assert mod.audit(system)["ok"]


# 兜底 / 确定性 / 审计 ------------------------------------------------------


def test_audit_flags_leftover_traces_on_dry_ground():
    system, surface = fresh_system(wetness=0.0)
    # 模拟"干燥后没清干净"的坏状态。
    surface.footprints.append(
        mod.Footprint(surface=surface.name, position=(0.0,), wetness=0.8, tick=0)
    )
    surface.add_decal("wet_slick")

    report = mod.audit(surface, system=system)

    assert report["ok"] is False
    assert any("脚印" in item for item in report["violations"])
    assert any("贴花" in item for item in report["violations"])


def test_determinism_and_frame_split():
    result = mod.determinism()
    assert result["ok"]
    assert result["identical"]
    assert result["frame_rate_split_ok"]


def test_fallback_never_returns_nan():
    for bad in (float("nan"), float("inf"), -5.0, "unknown", {"material": "lava", "wetness": "x"}):
        payload = mod.fallback(bad)
        assert payload["degraded"] is True
        assert math.isfinite(payload["physics_friction"])
        assert math.isfinite(payload["roughness"])
        assert 0.0 <= payload["wetness"] <= 1.0


def test_metrics_reports_all_surfaces():
    system, _ = fresh_system()
    report = mod.metrics(system)
    assert report["tick_seconds"] == pytest.approx(1.0 / 30.0)
    assert "ground" in report["surfaces"]
