"""Small general helpers with real callers (hypso.load.utils imports
is_integer_num)."""


def is_integer_num(n) -> bool:
    """
    Check if a number is an integer

    :param n: Number to check
    :return: Boolean value indicating whether or not the number is an integer
    """
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False
