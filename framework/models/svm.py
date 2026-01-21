from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')

class SVMModel:
    def __init__(self, hyperparameters=None, C=0.1):
        """Initialize SVM model with hyperparameters"""
        # Use LinearSVC for better performance on large datasets
        params = {
            'C': C,
            'random_state': 42,
            'max_iter': 2000,
            'dual': False,
            'loss': 'squared_hinge',
            'verbose': 0
        }
        
        self.model = LinearSVC(**params)
        self.hyperparameters = {'C': C}