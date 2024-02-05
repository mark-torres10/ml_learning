"""My implementation of the AdaBoost algorithm.

Inspired by https://www.youtube.com/watch?v=LsK-xG1cLYA
"""
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target


class AdaBoost:
    def __init__(
        self,
        clf: DecisionTreeClassifier,
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int = 100
    ):
        self.base_clf = clf
        self.clfs_list = []
        self.clfs_errors = []
        self.X = X
        self.y = y
        self.data = np.column_stack((X, y))
        self.sample_weights = np.ones(len(X)) / len(X) # init as equal sample weights
        self.n_iterations = n_iterations


    def train(self):
        """Train the AdaBoost classifier.
        
        The algorithm is as follows:
        1. Initialize the sample weights as equal
        2. For each iteration:
            a. Sample the data based on the sample weights
            b. Fit a classifier on the sample
            c. Get predictions and accuracies
            d. Calculate the error for the classifier
            e. Update the sample weights to emphasize the misclassified samples
            f. Normalize the sample weights
        """
        for _ in range(self.n_iterations):
            # sample self.data based on self.sample_weights
            sample = self.data[
                np.random.choice(
                    self.data.shape[0],
                    len(self.data),
                    p=self.sample_weights
                )
            ]
            X_sample = sample[:, :-1]
            y_sample = sample[:, -1]
            # fit classifier on sample
            clf = self.base_clf(max_depth=1)
            clf.fit(X_sample, y_sample)
            self.clfs_list.append(clf)
            # get predictions and accuracies
            y_pred = clf.predict(self.X)
            # error for a stump is equal to the sum of the weights of the
            # misclassified samples
            error = np.sum(self.sample_weights * (y_pred != self.y))
            self.clfs_errors.append(error)

            # re-weight the samples to emphasize the misclassified samples
            # Note: max(error, 1e-16) is used to avoid division by zero
            alpha = 0.5 * np.log((1 - error) / max(error, 1e-16))
            
            # Update sample weights
            # Misclassified samples get increased weights, correctly classified get decreased
            self.sample_weights *= np.exp(-alpha * y_sample * y_pred)

            # Normalize the sample weights
            self.sample_weights /= np.sum(self.sample_weights)


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # Compute the weighted vote of each classifier
        clf_votes = np.array(
            [
                alpha * clf.predict(X_test)
                for alpha, clf in zip(self.clfs_errors, self.clfs_list)
            ]
        )
        # Sum the votes across all classifiers for each sample
        y_pred = np.sign(np.sum(clf_votes, axis=0))
        return y_pred


if __name__ == "__main__":
    # init with a stump
    clf = DecisionTreeClassifier
    adabost = AdaBoost(clf, X, y)
    adabost.train()
    y_pred = adabost.predict(X)
    accuracy = np.mean(y_pred == y)
    print(f"Accuracy: {accuracy:.2f}")
