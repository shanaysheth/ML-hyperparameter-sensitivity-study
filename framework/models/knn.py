from sklearn.neighbors import KNeighborsClassifier

class KNNModel:
    def __init__(self, hyperparameters=None, n_neighbors=5):
        """Initialize KNN model with hyperparameters"""
        # Set default hyperparameters
        params = {
            'n_neighbors': n_neighbors,
            'metric': 'euclidean',
            'n_jobs': -1
        }
        
        self.model = KNeighborsClassifier(**params)
        self.hyperparameters = {'n_neighbors': n_neighbors}

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)