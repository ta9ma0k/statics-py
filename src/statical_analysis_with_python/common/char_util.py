def make_alphabet(length: int) -> list[str]:
    if length > 26:
        raise Exception('out of index.')
    return [chr(ord("A") + i) for i in range(length)]
