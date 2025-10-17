import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithms.hierarchical import HierarchicalClustering
from utils.data_loader import DataGenerator
from utils.visualizer import Visualizer
from utils.evaluator import Evaluator
import matplotlib.pyplot as plt

def run_hierarchical_example():
    """Exemple complet d'utilisation de la CAH"""
    print("=== EXEMPLE CLASSIFICATION ASCENDANTE HIÉRARCHIQUE ===")
    
    # Génération des données
    X, y_true = DataGenerator.generate_clustering_data(n_samples=100)  # Réduit pour la démo
    
    # Clustering hiérarchique
    cah = HierarchicalClustering(method='ward')
    linkage_matrix = cah.fit(X)
    
    # Obtention des clusters pour différents k
    labels_2 = cah.get_clusters(2)
    labels_4 = cah.get_clusters(4)
    
    # Évaluation
    metrics_4 = Evaluator.evaluate_clustering(X, y_true, labels_4, "CAH (k=4)")
    
    # Visualisations
    viz = Visualizer()
    
    # Dendrogramme
    cah.plot_dendrogram("CAH - Dendrogramme")
    
    # Comparaison des clusters
    y_list = [y_true, labels_2, labels_4]
    titles = ['Vrais Clusters', 'CAH - 2 Clusters', 'CAH - 4 Clusters']
    viz.plot_comparison(X, y_list, titles)
    
    return cah, metrics_4

if __name__ == "__main__":
    run_hierarchical_example()