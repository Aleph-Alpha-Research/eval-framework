from eval_framework.metrics.aggregators.aggregators import closed_form_passatk


def test_closed_form_passatk():
    assert closed_form_passatk(4, 2, 1) == 1 - 2 / 4
    assert closed_form_passatk(4, 2, 2) == 1 - 1 / 6
    assert closed_form_passatk(4, 2, 3) == 1
    assert closed_form_passatk(4, 2, 4) == 1.0
    assert closed_form_passatk(10, 3, 2) == 1 - 21 / 45
