from mini_ignition.simulator import ADD, HALT, LOAD, MOV, MUL, STORE, Program, ToyDevice


def run_program(*instructions):
    device = ToyDevice(memory_size=64)
    device.run(Program.from_instructions(*instructions))
    return device


def test_mov_works():
    device = run_program(MOV("r0", 7.5), HALT())

    assert device.read_scalar_register("r0") == 7.5


def test_add_works():
    device = run_program(MOV("r0", 2.0), MOV("r1", 3.0), ADD("r2", "r0", "r1"), HALT())

    assert device.read_scalar_register("r2") == 5.0
    assert device.arithmetic_op_count == 1


def test_mul_works():
    device = run_program(MOV("r0", 6.0), MOV("r1", 7.0), MUL("r2", "r0", "r1"), HALT())

    assert device.read_scalar_register("r2") == 42.0


def test_load_works():
    device = ToyDevice(memory_size=64)
    device.memory.write_scalar(4, 12.25)

    device.run(Program.from_instructions(LOAD("r0", 4), HALT()))

    assert device.read_scalar_register("r0") == 12.25


def test_store_works():
    device = run_program(MOV("r0", 9.0), STORE("r0", 4), HALT())

    assert device.memory.read_scalar(4) == 9.0


def test_halt_stops_execution():
    device = run_program(MOV("r0", 1.0), HALT(), MOV("r0", 2.0))

    assert device.halted is True
    assert device.read_scalar_register("r0") == 1.0
    assert device.instruction_count == 2
