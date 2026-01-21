from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

class LogisticRegressionModel:
    def __init__(self, hyperparameters=None, max_iter=2000, C=1.0):
        """Initialize Logistic Regression model with hyperparameters"""
        # Set default hyperparameters
        params = {
            'max_iter': max_iter,
            'C': C,
            'random_state': 42,
            'solver': 'lbfgs',
            'n_jobs': -1,
            'verbose': 0
        }
        
        self.model = LogisticRegression(**params)
        self.hyperparameters = {'max_iter': max_iter, 'C': C}

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)