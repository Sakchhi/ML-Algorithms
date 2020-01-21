import numpy as np


def get_coeffs(x, y):
    """
    Returns coefficients b1, b0 of an equation of the form y = b1.x + b0
    which is the least squares solution providing the line of best fit
    :param x: numpy array, representing independent features
    :param y: numpy array, representing dependent feature
    :return:
    """
    Sxy = np.matmul((x - np.mean(x)), (y - np.mean(y)))
    Sxx = np.square((x - np.mean(x))).sum()

    b1 = Sxy/Sxx
    b0 = np.mean(y) - b1*np.mean(x)
    return b1, b0


if __name__ == '__main__':
    x = np.random.randn(100)
    y = [5*i + 4 for i in x]
    print(get_coeffs(x, y))
