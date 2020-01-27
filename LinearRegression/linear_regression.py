import numpy as np
from sklearn import metrics


class LinearRegression:

    def __init__(self, fit_intercept=True):
        self.coefs_ = None
        self.intercept_ = None
        self.fit_intercept = fit_intercept

    def add_constant(self, X):
        return np.c_[X, np.ones(X.shape[0])]

    def fit(self, X, y):
        """
        Returns coefficients b1, b0 of an equation of the form y = b1.x + b0
        which is the least squares solution providing the line of best fit
        :param x: numpy array, representing independent features
        :param y: numpy array, representing dependent feature
        :return:
        """
        if self.fit_intercept:
            X = self.add_constant(X)
        b = (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)
        self.coefs_ = b[:-1]
        self.intercept_ = b[-1]
        return self

    def predict(self, test):
        """
        Return predictions for fitted model
        :param test:
        :return:
        """
        if not all(self.coefs_):
            raise ValueError("Fit the model before predicting")
        predictions = test.dot(self.coefs_) + self.intercept_
        return predictions

    def score(self, actuals, predictions):
        """
        Returns R-squared score for the fitted model
        :param actuals: Matrix of dependent feature
        :param predictions: Matrix of predictions
        :return: R-squared score
        """
        rss = np.square(predictions-actuals).sum()
        tss = np.square(actuals - actuals.mean()).sum()
        r2 = 1 - rss/tss
        return r2


if __name__ == '__main__':
    x = np.random.rand(100, 5)
    actual_coeffs = np.array([4.5, 5.2, 3.7, 9.1, 6.8])
    y = np.array((x * actual_coeffs).sum(axis=1) + 5).reshape(-1, 1)
    print(y.shape, x.shape)
    linear_reg = LinearRegression()
    linear_reg.fit(x, y)
    x_test = np.random.rand(10, 5)
    y_test = np.array((x_test * actual_coeffs).sum(axis=1) + 5).reshape(-1, 1)
    print(linear_reg.coefs_)
    y_pred = linear_reg.predict(x_test)
    err = (abs(y_test - y_pred)/y_test).mean()
    print(err)
    actual_r2 = metrics.r2_score(y_test, y_pred)
    our_r2 = linear_reg.score(y_test, y_pred)
    print(actual_r2, our_r2)