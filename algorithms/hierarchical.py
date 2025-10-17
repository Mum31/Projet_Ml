import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

class HierarchicalClustering:
    """Implémentation de la Classification Ascendante Hiérarchique"""
    
    def __init__(self, method='ward', metric='euclidean'):
        self.method = method
        self.metric = metric
        self.linkage_matrix = None
    
    def fit(self, X, n_clusters=None):
        """
        Effectue le clustering hiérarchique
        
        Parameters:
        X (array): Données à clusteriser
        n_clusters (int): Nombre de clusters souhaité (optionnel)
        
        Returns:
        array: Labels des clusters si n_clusters est spécifié, sinon la matrice de liaison
        """
        self.linkage_matrix = linkage(X, method=self.method, metric=self.metric)
        
        if n_clusters is not None:
            return self.get_clusters(n_clusters)
        
        return self.linkage_matrix
    
    def get_clusters(self, n_clusters):
        """
        Obtient les clusters pour un nombre donné
        
        Parameters:
        n_clusters (int): Nombre de clusters
        
        Returns:
        array: Labels des clusters
        """
        if self.linkage_matrix is None:
            raise ValueError("Le modèle doit être entraîné avant de prédire")
        
        return fcluster(self.linkage_matrix, n_clusters, criterion='maxclust') - 1
    
    def plot_dendrogram(self, title="Dendrogramme CAH"):
        """
        Trace le dendrogramme
        """
        if self.linkage_matrix is None:
            raise ValueError("Le modèle doit être entraîné avant de tracer")
        
        plt.figure(figsize=(12, 8))
        dendrogram(self.linkage_matrix)
        plt.title(title)
        plt.xlabel('Indices des échantillons')
        plt.ylabel('Distance')
        plt.show()