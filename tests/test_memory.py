import pytest

from mini_ignition.simulator.memory import Memory, MemoryAccessError, MemoryAlignmentError


def test_scalar_read_write_works():
    memory = Memory(size=16)

    memory.write_scalar(3, 1.5)

    assert memory.read_scalar(3) == 1.5


def test_vector_read_write_works():
    memory = Memory(size=16)

    memory.write_vector(2, [1.0, 2.0, 3.0, 4.0])

    assert memory.read_vector(2, 4).tolist() == [1.0, 2.0, 3.0, 4.0]


def test_out_of_bounds_access_fails():
    memory = Memory(size=8)

    with pytest.raises(MemoryAccessError, match="out of bounds"):
        memory.read_vector(6, 3)


def test_misaligned_access_fails_when_alignment_checking_is_enabled():
    memory = Memory(size=16, alignment=4, check_alignment=True)

    with pytest.raises(MemoryAlignmentError, match="not aligned"):
        memory.write_scalar(2, 99.0)
