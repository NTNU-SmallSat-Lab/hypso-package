"""Small general helpers."""


def is_integer_num(n) -> bool:
    """
    Check if a number is an integer

    Parameters
    ----------
    n
        Number to check

    Returns
    -------
    out
        Boolean value indicating whether or not the number is an integer
    """
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False
