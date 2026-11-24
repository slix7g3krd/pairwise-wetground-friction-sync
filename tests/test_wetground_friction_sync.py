import pytest
import gameplay.wetground_friction_sync as mod

CASES = [
    ("tick_accumulate", (1,)),
    ("shared_state", (1,)),
    ("decay_by_material", (1,)),
    ("release_decal", (1,)),
    ("sfx_current_state", (1,)),
    ("rain", (1,)),
    ("accumulate", (1,)),
    ("apply_friction", (1,)),
    ("render", (1,)),
    ("dry", (1,)),
    ("validate_grip", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
