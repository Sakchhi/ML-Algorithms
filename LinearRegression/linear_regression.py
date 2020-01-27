import numpy as np


class LinearRegression:

    def __init__(self, fit_intercept=True):
        self.coefs_ = np.nan
        self.intercept_ = np.nan




def fit(x, y):
    """
    Returns coefficients b1, b0 of an equation of the form y = b1.x + b0
    which is the least squares solution providing the line of best fit
    :param x: numpy array, representing independent features
    :param y: numpy array, representing dependent feature
    :return:
    """
    print((x - x.mean()).shape)
    Sxy = np.matmul((x - x.mean(axis=0)), (y - y.mean(axis=0)))
    Sxx = np.square((x - np.mean(x))).sum()

    b1 = Sxy/Sxx
    b0 = np.mean(y) - b1*np.mean(x)
    return b1, b0


if __name__ == '__main__':
    x = np.random.rand(100, 5)
    actual_coeffs = np.array([4.5, 5.2, 3.7, 9.1, 6.8])
    y = np.array((x*actual_coeffs).sum(axis=1) + 5).reshape(-1, 1)
    print(y.shape, x.shape)
    print(fit(x, y))
