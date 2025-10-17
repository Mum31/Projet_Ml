import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from algorithms.knn import KNN
from algorithms.kmeans import KMeans
from algorithms.hierarchical import HierarchicalClustering
from utils.data_loader import DataGenerator

class TestAlgorithms(unittest.TestCase):
    
    def setUp(self):
        """Setup pour les tests"""
        # Génération des données COMPLÈTES (pas seulement l'entraînement)
        X, y = DataGenerator.generate_classification_data_unsplit(n_samples=100)
        self.X_class = X
        self.y_class = y
        
        self.X_cluster, self.y_cluster = DataGenerator.generate_clustering_data(n_samples=100)
    
    def test_knn_initialization(self):
        """Test de l'initialisation de KNN"""
        knn = KNN(k=5)
        self.assertEqual(knn.k, 5)
    
    def test_knn_prediction(self):
        """Test des prédictions de KNN"""
        knn = KNN(k=3)
        # Utilisation des 80 premières samples pour l'entraînement
        knn.fit(self.X_class[:80], self.y_class[:80])
        # Prédiction sur les 20 dernières samples
        predictions = knn.predict(self.X_class[80:])
        self.assertEqual(len(predictions), 20)
    
    def test_kmeans_initialization(self):
        """Test de l'initialisation de K-Means"""
        kmeans = KMeans(k=4)
        self.assertEqual(kmeans.k, 4)
    
    def test_kmeans_clustering(self):
        """Test du clustering K-Means"""
        kmeans = KMeans(k=3, max_iters=10)
        labels = kmeans.fit(self.X_cluster)
        self.assertEqual(len(labels), len(self.X_cluster))
        # Note: K-Means peut créer moins de clusters que k si l'initialisation est mauvaise
        self.assertLessEqual(len(np.unique(labels)), 3)
    
    def test_hierarchical_initialization(self):
        """Test de l'initialisation de la CAH"""
        cah = HierarchicalClustering(method='ward')
        self.assertEqual(cah.method, 'ward')

if __name__ == '__main__':
    unittest.main()