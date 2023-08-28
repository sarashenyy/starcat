import matplotlib.pyplot as plt

from starcat.synstars import round_to_step


def test_round_to_step(arr, step):
    """Draw round result."""
    round_result = round_to_step(arr, step)
    plt.scatter(arr, round_result)
    plt.title(f"step = {step}")
    plt.xlabel('before round')
    plt.ylabel('after round to step')
    plt.show()
