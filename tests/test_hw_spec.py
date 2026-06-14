from mini_ignition.schemas.hw_spec import HardwareSpec


def test_hardware_spec_save_load_works(tmp_path):
    spec = HardwareSpec(
        name="toy-accelerator-test",
        memory_size=1024,
        vector_width=4,
        num_scalar_registers=8,
        num_vector_registers=8,
        alignment=4,
        supported_opcodes=["MOV", "ADD", "HALT"],
        latencies={"MOV": 1, "ADD": 1, "HALT": 1},
        has_vector_ops=False,
        has_dot=False,
        notes=["round-trip test"],
    )
    path = tmp_path / "hw_spec.json"

    spec.to_json(path)
    loaded = HardwareSpec.from_json(path)

    assert loaded == spec
