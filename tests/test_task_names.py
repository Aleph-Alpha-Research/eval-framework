from eval_framework.task_names import TaskName


def test_taskname_case_insensitive_lookup() -> None:
    math_task = TaskName.from_name("math")
    assert math_task == TaskName.MATH
    math_task2 = TaskName.from_name("MATH")
    assert math_task2 == TaskName.MATH
    math_task3 = TaskName.from_name("Math")
    assert math_task3 == TaskName.MATH
    mathlvl5_task = TaskName.from_name("math lvl 5")
    assert mathlvl5_task.value.NAME == "Math Lvl 5"
    mathlvl5_task2 = TaskName.from_name("MATH LVL 5")
    assert mathlvl5_task2.value.NAME == "Math Lvl 5"
    mathlvl5_task3 = TaskName.from_name("Math Lvl 5")
    assert mathlvl5_task3.value.NAME == "Math Lvl 5"
