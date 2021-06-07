import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# stump decyzyjny - weak classifier
class DecisionStump():
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions


class Adaboost(BaseEstimator, ClassifierMixin):

    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.y_ = y

        self.y_[self.y_ == 0] = -1

        n_samples, n_features = X.shape

        # Inicjalizacja wag
        w = np.full(n_samples, (1 / n_samples))

        self.clfs_ = []
        
        for _ in range(self.n_clf):
            clf = DecisionStump()

            min_error = float('inf')
            
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    # Error = suma wag błędnie sklasyfikowanych wzorców
                    misclassified = w[self.y_ != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # zachowanie najlepszego rezultatu
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            predictions = clf.predict(X)

            w *= np.exp(-clf.alpha * self.y_ * predictions)
            w /= np.sum(w)

            self.clfs_.append(clf)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs_]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        y_pred[y_pred == -1] = 0
        return y_pred