import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithms.kmeans import KMeans
from utils.data_loader import DataGenerator
from utils.visualizer import Visualizer
from utils.evaluator import Evaluator

def run_kmeans_example():
    """Exemple complet d'utilisation de K-Means"""
    print("=== EXEMPLE K-MEANS ===")
    
    # Génération des données
    X, y_true = DataGenerator.generate_clustering_data()
    
    # Clustering
    kmeans = KMeans(k=4)
    labels = kmeans.fit(X)
    
    # Évaluation
    metrics = Evaluator.evaluate_clustering(X, y_true, labels, "K-Means")
    
    # Visualisation
    viz = Visualizer()
    viz.plot_clustering_results(X, labels, kmeans.centroids, "K-Means - Résultats")
    
    # Comparaison avec les vrais clusters
    y_list = [y_true, labels]
    titles = ['Vrais Clusters', 'K-Means Clusters']
    viz.plot_comparison(X, y_list, titles)
    
    return kmeans, metrics

if __name__ == "__main__":
    run_kmeans_example()