from sklearn.tree import DecisionTreeClassifier

class DecisionTreeModel:
    def __init__(self, hyperparameters=None, max_depth=5, min_samples_split=2):
        """Initialize Decision Tree model with hyperparameters"""
        # Set default hyperparameters
        params = {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
        self.model = DecisionTreeClassifier(**params)
        self.hyperparameters = {'max_depth': max_depth, 'min_samples_split': min_samples_split}

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)