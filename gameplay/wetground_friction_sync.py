"""湿滑地面与摩擦同步。

对应六个修复点：

1. ``fixed tick``：湿度只在固定长度的模拟 tick 上累积，渲染帧率只决定一次调用
   里推进多少个 tick，不会改变同一段真实时间内的总增湿量。
2. ``shared_state``：物理摩擦系数与渲染粗糙度由同一份 ``wetness`` 快照
   （``SharedState``）一次性派生，二者不可能取到不同帧/不同来源的湿度。
3. ``decay_by_material``：雨停后的衰减速率按材质分档（水洼/沥青慢、金属/水泥中、
   草地/泥土快），不会所有材质同时变干。
4. ``release_decal`` / ``set_material``：材质切换时释放旧材质的湿滑贴花，切回
   干燥后不会再有旧贴花叠在路面上。
5. ``sfx_current_state`` / ``step_footstep``：脚步音效读取当帧的湿度快照，
   不再使用上一帧缓存，踩水声与脚印同帧。
6. ``dry`` / ``_step_surface``：湿度归零时清除脚印并释放湿滑贴花。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

# 固定模拟 tick，与渲染帧率解耦。30Hz 足够湿滑玩法，且跨机器一致。
DEFAULT_TICK_SECONDS = 1.0 / 30.0

_TICK_EPS = 1e-9  # 浮点累加器判定的容差
_DRY_EPS = 1e-6   # 低于该湿度视为完全干燥
RAIN_GAIN_PER_SECOND = 0.9  # 雨强 1.0、吸水率 1.0 时的最大增湿速度（每秒）
WET_SFX_THRESHOLD = 0.15    # 高于该湿度播放踩水声
FOOTPRINT_THRESHOLD = 0.10  # 高于该湿度留下湿脚印


def _clamp01(value: float) -> float:
    """把任意输入夹到 [0, 1]，非有限值一律当作 0。"""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


# --------------------------------------------------------------------------- 材质


@dataclass(frozen=True)
class Material:
    """一种地面材质的物理/视觉/蒸发参数。"""

    key: str
    dry_friction: float
    wet_friction: float
    roughness_dry: float
    roughness_wet: float
    decay_per_second: float  # 无雨时每秒蒸发的湿度
    absorbance: float        # 雨水吸收效率
    footstep: str = "footstep"

    @property
    def decay_tier(self) -> str:
        """衰减档位：slow / medium / fast。"""
        if self.decay_per_second <= 0.05:
            return "slow"
        if self.decay_per_second <= 0.15:
            return "medium"
        return "fast"


MATERIALS: Dict[str, Material] = {
    # 池水：几乎不蒸发，但本身很湿。
    "water": Material("water", 0.95, 0.45, 0.05, 0.02, 0.02, 0.95, "splash"),
    # 沥青：排水稍好、蒸发慢。
    "asphalt": Material("asphalt", 0.90, 0.55, 0.85, 0.35, 0.05, 0.50, "step_wet"),
    # 金属：吸水少，干得中等。
    "metal": Material("metal", 0.70, 0.30, 0.25, 0.08, 0.07, 0.15, "step_wet"),
    # 水泥：吸水多、蒸发中等，是本场景的主地面。
    "concrete": Material("concrete", 0.85, 0.38, 0.78, 0.28, 0.12, 0.70, "step_wet"),
    # 泥土 / 草地：吸得快也干得快。
    "dirt": Material("dirt", 0.75, 0.42, 0.88, 0.50, 0.22, 0.80, "step_damp"),
    "grass": Material("grass", 0.80, 0.50, 0.90, 0.60, 0.30, 0.55, "step_damp"),
}

DEFAULT_MATERIAL = "concrete"


def as_material(value) -> Material:
    """把材质名 / Surface / Material 统一成 Material。"""
    if isinstance(value, Material):
        return value
    if isinstance(value, Surface):
        return value.material
    if isinstance(value, str):
        return MATERIALS.get(value, MATERIALS[DEFAULT_MATERIAL])
    return MATERIALS[DEFAULT_MATERIAL]


# ----------------------------------------------------------------------- 状态对象


@dataclass
class Decal:
    """一块湿滑贴花，带显式释放语义。"""

    surface_name: str
    kind: str = "wet_slick"
    material: str = DEFAULT_MATERIAL
    released: bool = False
    release_count: int = 0

    def release(self) -> bool:
        """幂等释放；首次释放返回 True。"""
        if self.released:
            return False
        self.released = True
        self.release_count += 1
        return True


@dataclass
class Footprint:
    """一个湿脚印，记录留下它时的湿度与 tick。"""

    surface: str
    position: Tuple[float, ...]
    wetness: float
    tick: int


@dataclass(frozen=True)
class SharedState:
    """同一帧内物理与渲染共同读取的唯一湿度快照。"""

    material: str
    wetness: float
    version: int
    physics_friction: float
    roughness: float
    decay_per_second: float
    decay_tier: str


def derive_state(material, wetness, version: int = 0) -> SharedState:
    """物理与渲染的唯一派生入口：同源快照。"""
    mat = as_material(material)
    wet = _clamp01(wetness)
    friction = mat.dry_friction + (mat.wet_friction - mat.dry_friction) * wet
    roughness = mat.roughness_dry + (mat.roughness_wet - mat.roughness_dry) * wet
    return SharedState(
        material=mat.key,
        wetness=wet,
        version=int(version),
        physics_friction=round(friction, 9),
        roughness=round(roughness, 9),
        decay_per_second=mat.decay_per_second,
        decay_tier=mat.decay_tier,
    )


@dataclass
class Surface:
    """一块地面：湿度是唯一真值，物理/渲染/音效都从它派生。"""

    name: str
    material: Material
    wetness: float = 0.0
    version: int = 0
    decals: List[Decal] = field(default_factory=list)
    footprints: List[Footprint] = field(default_factory=list)

    # 观测用的版本戳，用于审计"读取了哪一帧"。
    last_physics_version: int = -1
    last_render_version: int = -1
    last_sfx_version: int = -1

    released_decals: int = 0
    cleared_footprints: int = 0
    material_switches: int = 0

    def snapshot(self) -> SharedState:
        return derive_state(self.material, self.wetness, self.version)

    def live_decals(self) -> List[Decal]:
        return [d for d in self.decals if not d.released]

    def add_decal(self, kind: str = "wet_slick") -> Decal:
        decal = Decal(surface_name=self.name, kind=kind, material=self.material.key)
        self.decals.append(decal)
        return decal


def _sync_wet_decals(surface: Surface) -> None:
    """湿度 > 0 时维持一块湿滑贴花；归零时全部释放。"""
    if surface.wetness > 0.0:
        if not surface.live_decals():
            surface.add_decal("wet_slick")
        return
    for decal in list(surface.decals):
        if decal.release():
            surface.released_decals += 1
        surface.decals.remove(decal)


def _step_surface(surface: Surface, dt: float, rain_intensity: float) -> bool:
    """按固定 tick 推进一块地面。返回湿度是否变化。"""
    mat = surface.material
    gain = _clamp01(rain_intensity) * RAIN_GAIN_PER_SECOND * mat.absorbance * dt
    loss = mat.decay_per_second * dt
    wetness = _clamp01(surface.wetness + gain - loss)
    if wetness < _DRY_EPS:
        wetness = 0.0
    changed = wetness != surface.wetness
    surface.wetness = wetness
    if changed:
        surface.version += 1

    _sync_wet_decals(surface)
    if surface.wetness == 0.0 and surface.footprints:
        surface.cleared_footprints += len(surface.footprints)
        surface.footprints.clear()
    return changed


# --------------------------------------------------------------------------- 系统


class GroundSystem:
    """湿滑地面系统：固定 tick 时钟 + 一组地面。"""

    def __init__(self, tick_seconds: float = DEFAULT_TICK_SECONDS):
        if tick_seconds <= 0:
            raise ValueError("tick_seconds 必须为正数")
        self.tick_seconds = float(tick_seconds)
        self.ticks = 0
        self.rain_intensity = 0.0
        self.accumulated = 0.0
        self.surfaces: Dict[str, Surface] = {}
        self._default_name: Optional[str] = None

    # -- 地面管理 ---------------------------------------------------------
    def add_surface(self, name: str, material=DEFAULT_MATERIAL, wetness: float = 0.0) -> Surface:
        surface = Surface(name=name, material=as_material(material), wetness=_clamp01(wetness))
        self.surfaces[name] = surface
        if self._default_name is None:
            self._default_name = name
        return surface

    def default_surface(self) -> Surface:
        if self._default_name is None:
            self.add_surface("ground", DEFAULT_MATERIAL)
        return self.surfaces[self._default_name]

    def surface(self, name: str) -> Surface:
        if name not in self.surfaces:
            self.add_surface(name, name)
        return self.surfaces[name]

    def set_rain(self, intensity: float) -> float:
        self.rain_intensity = _clamp01(intensity)
        return self.rain_intensity

    # -- 固定 tick 推进 ---------------------------------------------------
    def fixed_steps(self, seconds: float) -> int:
        """一段真实时间对应多少个完整 tick（向下取整）。"""
        if seconds is None or seconds <= 0:
            return 0
        return int(math.floor(float(seconds) / self.tick_seconds + _TICK_EPS))

    def tick(self) -> None:
        self.ticks += 1
        for surface in self.surfaces.values():
            _step_surface(surface, self.tick_seconds, self.rain_intensity)

    def accumulate(self, seconds: float, rain: Optional[float] = None) -> dict:
        """累积真实时间，只在固定 tick 边界推进模拟。"""
        if rain is not None:
            self.set_rain(rain)
        if seconds is None or seconds <= 0:
            return {
                "ticks": 0,
                "remainder": self.accumulated,
                "clock": self.ticks,
                "rain": self.rain_intensity,
            }
        self.accumulated += float(seconds)
        steps = int(math.floor(self.accumulated / self.tick_seconds + _TICK_EPS))
        self.accumulated -= steps * self.tick_seconds
        if self.accumulated < 0.0:
            self.accumulated = 0.0
        for _ in range(steps):
            self.tick()
        return {
            "ticks": steps,
            "remainder": self.accumulated,
            "clock": self.ticks,
            "rain": self.rain_intensity,
        }

    def dry_surface(self, surface: Surface, seconds: float) -> int:
        """对单块地面按固定 tick 蒸发 seconds 秒（无雨）。"""
        steps = self.fixed_steps(seconds)
        for _ in range(steps):
            _step_surface(surface, self.tick_seconds, 0.0)
        return steps

    # -- 材质 / 贴花 / 音效 ----------------------------------------------
    def set_material(self, surface, material) -> Surface:
        """切换材质：释放旧材质的湿滑贴花与脚印。"""
        if not isinstance(surface, Surface):
            surface = self.surface(str(surface))
        mat = as_material(material)
        if mat.key == surface.material.key:
            return surface

        for decal in list(surface.decals):
            if decal.release():
                surface.released_decals += 1
            surface.decals.remove(decal)
        if surface.footprints:
            surface.cleared_footprints += len(surface.footprints)
            surface.footprints.clear()

        surface.material = mat
        surface.version += 1
        surface.material_switches += 1
        # 不在切换瞬间重建贴花：下一次模拟 tick 若仍湿，会为新材质重新生成。
        return surface

    def release_decal(self, surface, decal: Optional[Decal] = None) -> dict:
        if not isinstance(surface, Surface):
            surface = self.surface(str(surface))
        if decal is not None:
            targets = [decal]
        else:
            targets = list(surface.decals)
        released = 0
        for item in targets:
            if not isinstance(item, Decal):
                continue
            if item.release():
                released += 1
                surface.released_decals += 1
            if item in surface.decals:
                surface.decals.remove(item)
        return {
            "surface": surface.name,
            "released": released,
            "live_decals": len(surface.live_decals()),
            "total_released": surface.released_decals,
        }

    def step_footstep(self, surface, position) -> dict:
        """在给定位置落下一步：音效与脚印都读当帧湿度。"""
        if not isinstance(surface, Surface):
            surface = self.surface(str(surface))
        state = surface.snapshot()
        surface.last_sfx_version = state.version

        has_footprint = state.wetness >= FOOTPRINT_THRESHOLD
        if has_footprint:
            if hasattr(position, "__iter__") and not isinstance(position, str):
                pos: Tuple[float, ...] = tuple(float(p) for p in position)
            else:
                pos = (float(position),)
            surface.footprints.append(
                Footprint(surface=surface.name, position=pos, wetness=state.wetness, tick=self.ticks)
            )

        return {
            "surface": surface.name,
            "position": position,
            "audio": "splash" if state.wetness >= WET_SFX_THRESHOLD else "dry",
            "volume": round(0.35 + 0.65 * state.wetness, 4),
            "wetness": state.wetness,
            "source_version": state.version,
            "footprint": has_footprint,
            "tick": self.ticks,
        }


def _build_default_system() -> GroundSystem:
    system = GroundSystem()
    system.add_surface("concrete", "concrete")
    system.add_surface("pool", "water")
    system.add_surface("asphalt", "asphalt")
    system.add_surface("grass", "grass")
    return system


DEFAULT_SYSTEM = _build_default_system()


# --------------------------------------------------------------------------- 内部


def _resolve_surface(surface=None, system=None):
    sys = system if isinstance(system, GroundSystem) else DEFAULT_SYSTEM
    if isinstance(surface, Surface):
        return sys, surface
    if isinstance(surface, str):
        return sys, sys.surface(surface)
    return sys, sys.default_surface()


def _sfx_payload(surface: Surface) -> dict:
    state = surface.snapshot()
    surface.last_sfx_version = state.version
    return {
        "surface": surface.name,
        "audio": "splash" if state.wetness >= WET_SFX_THRESHOLD else "dry",
        "volume": round(0.35 + 0.65 * state.wetness, 4),
        "wetness": state.wetness,
        "source_version": state.version,
        "current": True,
    }


def _system_or_default(value) -> GroundSystem:
    return value if isinstance(value, GroundSystem) else DEFAULT_SYSTEM


# ------------------------------------------------------------------ 公共契约入口


def tick_accumulate(dt: float = DEFAULT_TICK_SECONDS, *, system=None) -> dict:
    """按固定 tick 累积 dt 秒；返回本次推进的 tick 数与余量。"""
    return _system_or_default(system).accumulate(dt)


def accumulate(dt: float = DEFAULT_TICK_SECONDS, *, system=None, rain: Optional[float] = None) -> dict:
    """固定 tick 累积，附带各地面湿度快照。"""
    sys = _system_or_default(system)
    result = sys.accumulate(dt, rain=rain)
    result["surfaces"] = {name: surface.wetness for name, surface in sys.surfaces.items()}
    return result


def shared_state(surface=None, *, system=None) -> SharedState:
    """物理与渲染共同读取的同一份湿度快照。"""
    _, ground = _resolve_surface(surface, system)
    return ground.snapshot()


def decay_by_material(material=None, seconds: Optional[float] = None, *, system=None) -> dict:
    """报告各材质在无雨条件下的衰减：速率、档位、剩余湿度。"""
    if isinstance(material, (int, float)) and seconds is None:
        seconds, material = float(material), None
    seconds = DEFAULT_TICK_SECONDS if seconds is None else float(seconds)
    if isinstance(material, Surface):
        targets = [material.material]
    elif material is None:
        targets = list(MATERIALS.values())
    else:
        targets = [as_material(material)]

    report = {}
    for mat in targets:
        remaining = _clamp01(1.0 - mat.decay_per_second * seconds)
        report[mat.key] = {
            "decay_per_second": mat.decay_per_second,
            "tier": mat.decay_tier,
            "wetness_after": round(remaining, 9),
            "delta": round(1.0 - remaining, 9),
        }
    return {"dt": seconds, "materials": report}


def release_decal(surface=None, decal: Optional[Decal] = None, *, system=None) -> dict:
    """释放指定贴花；未指定则释放该地面全部贴花。"""
    sys = _system_or_default(system)
    if isinstance(surface, Decal) and decal is None:
        decal, surface = surface, None
    _, ground = _resolve_surface(surface, sys)
    return sys.release_decal(ground, decal)


def sfx_current_state(surface=None, *, system=None) -> dict:
    """脚步音效读取当帧状态，不缓存上一帧。"""
    _, ground = _resolve_surface(surface, system)
    return _sfx_payload(ground)


def rain(intensity: float = 1.0, *, system=None) -> dict:
    sys = _system_or_default(system)
    value = sys.set_rain(intensity)
    return {"rain": value, "clock": sys.ticks, "tick_seconds": sys.tick_seconds}


def apply_friction(surface=None, *, system=None) -> float:
    """物理摩擦系数，与渲染同源，湿润时真实降低。"""
    _, ground = _resolve_surface(surface, system)
    state = ground.snapshot()
    ground.last_physics_version = state.version
    return state.physics_friction


def render(surface=None, *, system=None) -> float:
    """渲染粗糙度，与物理摩擦读取同一份快照。"""
    _, ground = _resolve_surface(surface, system)
    state = ground.snapshot()
    ground.last_render_version = state.version
    return state.roughness


def dry(surface=None, seconds: Optional[float] = None, *, system=None) -> dict:
    """无雨蒸发指定地面若干秒（按固定 tick）。"""
    if isinstance(surface, (int, float)) and seconds is None:
        seconds, surface = float(surface), None
    sys = _system_or_default(system)
    _, ground = _resolve_surface(surface, sys)
    seconds = DEFAULT_TICK_SECONDS if seconds is None else float(seconds)
    steps = sys.dry_surface(ground, seconds)
    return {
        "surface": ground.name,
        "seconds": seconds,
        "ticks": steps,
        "wetness": ground.wetness,
        "footprints": len(ground.footprints),
        "live_decals": len(ground.live_decals()),
        "released_decals": ground.released_decals,
    }


def validate_grip(surface=None, *, system=None) -> dict:
    """校验物理与渲染确实同源，且湿度传导为单调。"""
    _, ground = _resolve_surface(surface, system)
    state = ground.snapshot()
    friction = apply_friction(ground)
    roughness = render(ground)

    issues: List[str] = []
    if not math.isclose(friction, state.physics_friction, abs_tol=1e-9):
        issues.append("物理摩擦与当前快照不一致")
    if not math.isclose(roughness, state.roughness, abs_tol=1e-9):
        issues.append("渲染粗糙度与当前快照不一致")
    if ground.last_physics_version != ground.last_render_version:
        issues.append("物理与渲染取用了不同版本的湿度")
    if ground.last_physics_version != ground.version:
        issues.append("物理摩擦读取了旧帧状态")
    if ground.last_render_version != ground.version:
        issues.append("渲染读取了旧帧状态")

    driest = derive_state(ground.material, 0.0)
    wettest = derive_state(ground.material, 1.0)
    if not (wettest.physics_friction - 1e-9 <= friction <= driest.physics_friction + 1e-9):
        issues.append("摩擦系数超出材质干湿区间")
    if not (wettest.roughness - 1e-9 <= roughness <= driest.roughness + 1e-9):
        issues.append("粗糙度超出材质干湿区间")

    return {
        "ok": not issues,
        "issues": issues,
        "surface": ground.name,
        "wetness": state.wetness,
        "physics_friction": friction,
        "roughness": roughness,
        "source_version": state.version,
        "physics_source_version": ground.last_physics_version,
        "render_source_version": ground.last_render_version,
    }


def metrics(system=None) -> dict:
    sys = _system_or_default(system)
    surfaces = {}
    for name, ground in sys.surfaces.items():
        state = ground.snapshot()
        surfaces[name] = {
            "material": state.material,
            "tier": state.decay_tier,
            "wetness": state.wetness,
            "physics_friction": state.physics_friction,
            "roughness": state.roughness,
            "live_decals": len(ground.live_decals()),
            "footprints": len(ground.footprints),
            "version": ground.version,
            "released_decals": ground.released_decals,
            "cleared_footprints": ground.cleared_footprints,
            "material_switches": ground.material_switches,
        }
    return {
        "tick_seconds": sys.tick_seconds,
        "ticks": sys.ticks,
        "rain": sys.rain_intensity,
        "accumulated": sys.accumulated,
        "surfaces": surfaces,
    }


def audit(surface=None, *, system=None) -> dict:
    """检查六条不变量是否成立，返回结构化违规列表。"""
    sys = system if isinstance(system, GroundSystem) else DEFAULT_SYSTEM
    if isinstance(surface, GroundSystem):
        sys, surface = surface, None
    targets = [_resolve_surface(surface, sys)[1]] if surface is not None else list(sys.surfaces.values())

    violations: List[str] = []
    if sys.tick_seconds <= 0:
        violations.append("固定 tick 必须为正数")

    for ground in targets:
        state = ground.snapshot()

        if state.wetness == 0.0:
            if ground.live_decals():
                violations.append(f"{ground.name}: 已干燥但仍残留湿滑贴花")
            if ground.footprints:
                violations.append(f"{ground.name}: 已干燥但仍残留脚印")
        if any(decal.released for decal in ground.decals):
            violations.append(f"{ground.name}: 已释放的贴花未从列表中移除")

        probe = _sfx_payload(ground)
        expected_audio = "splash" if state.wetness >= WET_SFX_THRESHOLD else "dry"
        if probe["source_version"] != ground.version:
            violations.append(f"{ground.name}: 音效读取了旧帧状态")
        if probe["audio"] != expected_audio:
            violations.append(f"{ground.name}: 音效与当帧湿度不一致")

        friction = apply_friction(ground)
        roughness = render(ground)
        if ground.last_physics_version != ground.last_render_version:
            violations.append(f"{ground.name}: 物理与渲染取用了不同版本的湿度")
        if not math.isclose(friction, state.physics_friction, abs_tol=1e-9):
            violations.append(f"{ground.name}: 物理摩擦并非来自当帧快照")
        if not math.isclose(roughness, state.roughness, abs_tol=1e-9):
            violations.append(f"{ground.name}: 渲染粗糙度并非来自当帧快照")

    return {
        "ok": not violations,
        "violations": violations,
        "surfaces": [ground.name for ground in targets],
        "ticks": sys.ticks,
        "tick_seconds": sys.tick_seconds,
    }


def fallback(state=None) -> dict:
    """状态损坏时的安全兜底：退回干燥材质，绝不把 NaN 传给物理。"""
    material_key = DEFAULT_MATERIAL
    wetness = 0.0

    if isinstance(state, SharedState):
        material_key, wetness = state.material, state.wetness
    elif isinstance(state, Surface):
        material_key, wetness = state.material.key, state.wetness
    elif isinstance(state, dict):
        material_key = state.get("material", DEFAULT_MATERIAL)
        wetness = state.get("wetness", 0.0)
    elif isinstance(state, (int, float)):
        wetness = state
    elif isinstance(state, str):
        material_key = state

    if not isinstance(material_key, str) or material_key not in MATERIALS:
        material_key = DEFAULT_MATERIAL
    derived = derive_state(material_key, wetness, version=-1)
    return {
        "material": derived.material,
        "wetness": derived.wetness,
        "physics_friction": derived.physics_friction,
        "roughness": derived.roughness,
        "degraded": True,
        "reason": "invalid_or_partial_state",
    }


def determinism(seed: int = 0, *, system=None) -> dict:
    """相同输入序列得到相同结果，且帧率拆分不改变结果。"""
    if isinstance(seed, GroundSystem):
        seed = 0

    frames = [1.0 / 30.0, 1.0 / 60.0, 1.0 / 60.0, 1.0 / 144.0, 1.0 / 144.0, 1.0 / 144.0, 0.25, 0.5]

    def run_once() -> list:
        sim = GroundSystem()
        for name in ("concrete", "pool", "asphalt", "grass"):
            sim.add_surface(name, name if name != "pool" else "water")
        sim.set_rain(0.7)
        for dt in frames:
            sim.accumulate(dt)
        for ground in sim.surfaces.values():
            sim.step_footstep(ground, (1.0, 0.0, 2.0))
        sim.set_rain(0.0)
        for dt in frames:
            sim.accumulate(dt)
        return [
            (
                name,
                round(ground.wetness, 12),
                ground.version,
                len(ground.footprints),
                len(ground.live_decals()),
            )
            for name, ground in sorted(sim.surfaces.items())
        ]

    first = run_once()
    second = run_once()

    # 1/30 一帧 与 1/60 两帧必须推进同样的 tick、得到同样的湿度。
    one = GroundSystem()
    one_ground = one.add_surface("floor", "concrete")
    one.set_rain(1.0)
    one.accumulate(1.0 / 30.0)

    split = GroundSystem()
    split_ground = split.add_surface("floor", "concrete")
    split.set_rain(1.0)
    split.accumulate(1.0 / 60.0)
    split.accumulate(1.0 / 60.0)

    split_ok = (
        one.ticks == split.ticks == 1
        and round(one_ground.wetness, 12) == round(split_ground.wetness, 12)
    )

    return {
        "ok": first == second and split_ok,
        "identical": first == second,
        "frame_rate_split_ok": split_ok,
        "signature": first,
        "seed": seed,
    }


def set_material(surface=None, material=DEFAULT_MATERIAL, *, system=None) -> dict:
    """切换地面材质并释放旧材质贴花。"""
    sys = _system_or_default(system)
    _, ground = _resolve_surface(surface, sys)
    sys.set_material(ground, material)
    return {
        "surface": ground.name,
        "material": ground.material.key,
        "live_decals": len(ground.live_decals()),
        "released_decals": ground.released_decals,
        "switches": ground.material_switches,
    }


def footstep(surface=None, position=(0.0, 0.0, 0.0), *, system=None) -> dict:
    """落下一步：当帧采样湿度，产生同帧音效与脚印。"""
    sys = _system_or_default(system)
    _, ground = _resolve_surface(surface, sys)
    return sys.step_footstep(ground, position)
