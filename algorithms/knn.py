import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

class KNN:
    """Implémentation de l'algorithme K-Nearest Neighbors"""
    
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        """
        Entraîne le modèle KNN
        
        Parameters:
        X (array): Features d'entraînement
        y (array): Labels d'entraînement
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """
        Prédit les labels pour les données de test
        
        Parameters:
        X (array): Features de test
        
        Returns:
        array: Labels prédits
        """
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        # Calcul des distances euclidiennes
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        
        # Obtention des k plus proches voisins
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # Vote majoritaire
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def score(self, X, y):
        """
        Calcule l'accuracy du modèle
        
        Parameters:
        X (array): Features de test
        y (array): Labels réels
        
        Returns:
        float: Accuracy
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)