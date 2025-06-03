import numpy as np

QAM16_points = {
    0: {'point': (-3, 3), 'bits': '0000'},
    1: {'point': (-3, 1), 'bits': '0001'},
    2: {'point': (-3, -1), 'bits': '0011'},
    3: {'point': (-3, -3), 'bits': '0010'},
    4: {'point': (-1, 3), 'bits': '0100'},
    5: {'point': (-1, 1), 'bits': '0101'},
    6: {'point': (-1, -1), 'bits': '0111'},
    7: {'point': (-1, -3), 'bits': '0110'},
    8: {'point': (1, 3), 'bits': '1100'},
    9: {'point': (1, 1), 'bits': '1101'},
    10: {'point': (1, -1), 'bits': '1111'},
    11: {'point': (1, -3), 'bits': '1110'},
    12: {'point': (3, 3), 'bits': '1000'},
    13: {'point': (3, 1), 'bits': '1001'},
    14: {'point': (3, -1), 'bits': '1011'},
    15: {'point': (3, -3), 'bits': '1010'},
}


def generate_original_symbols(num_symbols):
    index = np.random.randint(0, 16, num_symbols)
    original_symbols = np.array([QAM16_points[i]['point'] for i in index])
    bits = [QAM16_points[i]['bits'] for i in index]
    return original_symbols, bits


if __name__ == "__main__":
    original_symbols, bits = generate_original_symbols(1)
    print(original_symbols, bits)
