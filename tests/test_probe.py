from mini_ignition.probe.characterize import characterize_device
from mini_ignition.simulator import ToyDevice


def test_characterize_device_returns_correct_default_spec():
    spec = characterize_device(ToyDevice())

    assert spec.name == "toy-accelerator-v0"
    assert spec.memory_size == 4096
    assert spec.vector_width == 8
    assert spec.num_scalar_registers == 16
    assert spec.num_vector_registers == 16
    assert spec.alignment == 4
    assert spec.has_vector_ops is True
    assert spec.has_dot is True
    assert spec.latencies["MOV"] == 1
    assert spec.latencies["VLOAD"] == 4
    assert spec.latencies["DOT"] == 3


def test_supported_opcodes_include_core_scalar_vector_and_halt():
    spec = characterize_device(ToyDevice())

    assert {
        "MOV",
        "ADD",
        "MUL",
        "VLOAD",
        "VADD",
        "VMUL",
        "DOT",
        "HALT",
    }.issubset(set(spec.supported_opcodes))
