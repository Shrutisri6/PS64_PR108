# model.py
import numpy as np
from sklearn.linear_model import LogisticRegression

class MiningDetectionModel:
    def __init__(self):
        self.model = LogisticRegression()
        self.is_trained = False

    def extract_features(self, before, after):
        diff = np.abs(after - before)

        features = [
            np.mean(diff),
            np.std(diff),
            np.max(diff),
            np.sum(diff > 30) / diff.size
        ]

        return np.array(features).reshape(1, -1)

    def train_dummy(self):
        # Fake training data
        X = [
            [5, 2, 10, 0.01],
            [50, 20, 200, 0.4],
            [30, 10, 120, 0.2],
            [2, 1, 5, 0.005]
        ]
        y = [0, 1, 1, 0]

        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, before, after):
        if not self.is_trained:
            self.train_dummy()

        features = self.extract_features(before, after)
        prob = self.model.predict_proba(features)[0][1]

        return float(prob)
