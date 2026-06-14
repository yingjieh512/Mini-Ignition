import numpy as np

from mini_ignition.simulator import DOT, HALT, Program, ToyDevice, VADD, VLOAD, VMUL, VSTORE


def make_device():
    return ToyDevice(memory_size=64, vector_width=4)


def test_vload_works():
    device = make_device()
    device.memory.write_vector(0, [1.0, 2.0, 3.0, 4.0])

    device.run(Program.from_instructions(VLOAD("v0", 0), HALT()))

    np.testing.assert_allclose(device.read_vector_register("v0"), [1.0, 2.0, 3.0, 4.0])


def test_vstore_works():
    device = make_device()
    device.write_vector_register("v0", np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))

    device.run(Program.from_instructions(VSTORE("v0", 4), HALT()))

    np.testing.assert_allclose(device.memory.read_vector(4, 4), [5.0, 6.0, 7.0, 8.0])


def test_vadd_works():
    device = make_device()
    device.memory.write_vector(0, [1.0, 2.0, 3.0, 4.0])
    device.memory.write_vector(4, [10.0, 20.0, 30.0, 40.0])

    device.run(
        Program.from_instructions(
            VLOAD("v0", 0),
            VLOAD("v1", 4),
            VADD("v2", "v0", "v1"),
            HALT(),
        )
    )

    np.testing.assert_allclose(device.read_vector_register("v2"), [11.0, 22.0, 33.0, 44.0])


def test_vmul_works():
    device = make_device()
    device.memory.write_vector(0, [1.0, 2.0, 3.0, 4.0])
    device.memory.write_vector(4, [10.0, 20.0, 30.0, 40.0])

    device.run(
        Program.from_instructions(
            VLOAD("v0", 0),
            VLOAD("v1", 4),
            VMUL("v2", "v0", "v1"),
            HALT(),
        )
    )

    np.testing.assert_allclose(device.read_vector_register("v2"), [10.0, 40.0, 90.0, 160.0])


def test_dot_works():
    device = make_device()
    device.memory.write_vector(0, [1.0, 2.0, 3.0, 4.0])
    device.memory.write_vector(4, [10.0, 20.0, 30.0, 40.0])

    device.run(
        Program.from_instructions(
            VLOAD("v0", 0),
            VLOAD("v1", 4),
            DOT("r0", "v0", "v1"),
            HALT(),
        )
    )

    assert device.read_scalar_register("r0") == 300.0
