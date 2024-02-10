import numpy as np

def type_conversion_int(lines):
    return [int(item) for sublist in lines for item in sublist if item != ""]


def type_conversion_float(lines):
    return [float(item) for sublist in lines for item in sublist if item != ""]


def type_conversion_bool(lines):
    # return [bool(item) for sublist in lines for item in sublist if item != ""]
    return [True if item=='TRUE' else False for sublist in lines for item in sublist if item != ""]

def convert_file_to_series(file_name: str, shape, type_conversion):
    with open(r'C:\Users\dapole\OneDrive - Microsoft\masters\thesis\data\\' + file_name) as f:
        lines = f.readlines()
    lines = [line.replace("\n", "") for line in lines]
    lines = [line.replace(" ", "") for line in lines]
    lines = [line.split(",") for line in lines]
    lines = type_conversion(lines)
    return np.array(lines).reshape(shape[::-1]).T


def get_data():
    X = convert_file_to_series("X.txt", (3000, 12), type_conversion_float)
    Z = convert_file_to_series("Z.txt", (3000, 1), type_conversion_int)
    delta = convert_file_to_series("delta.txt", (3000, 12, 3), type_conversion_bool)
    return (X, Z, delta)


if __name__ == '__main__':
    X, Z, delta = get_data()