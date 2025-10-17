import numpy as np
from sklearn.datasets import make_blobs, make_classification, make_moons
from sklearn.model_selection import train_test_split

class DataGenerator:
    """Générateur de jeux de données synthétiques"""
    
    @staticmethod
    def generate_classification_data(n_samples=300, n_features=2, test_size=0.3, random_state=42):
        """
        Génère des données pour la classification avec split train/test
        """
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_features, 
            n_redundant=0, 
            n_informative=2,
            n_clusters_per_class=1, 
            random_state=random_state
        )
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    @staticmethod
    def generate_classification_data_unsplit(n_samples=300, n_features=2, random_state=42):
        """
        Génère des données pour la classification SANS split train/test
        (Pour les tests unitaires)
        """
        return make_classification(
            n_samples=n_samples, 
            n_features=n_features, 
            n_redundant=0, 
            n_informative=2,
            n_clusters_per_class=1, 
            random_state=random_state
        )
    
    @staticmethod
    def generate_clustering_data(n_samples=300, n_centers=4, cluster_std=0.60, random_state=42):
        """
        Génère des données pour le clustering
        """
        return make_blobs(
            n_samples=n_samples, 
            centers=n_centers, 
            cluster_std=cluster_std, 
            random_state=random_state
        )
    
    @staticmethod
    def generate_nonlinear_data(n_samples=300, noise=0.05, random_state=42):
        """
        Génère des données non-linéaires
        """
        return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    @staticmethod
    def load_iris_data():
        """
        Charge le dataset Iris
        """
        from sklearn.datasets import load_iris
        iris = load_iris()
        return iris.data, iris.target