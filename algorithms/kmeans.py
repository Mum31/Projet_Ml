import numpy as np

class KMeans:
    """Implémentation de l'algorithme K-Means"""
    
    def __init__(self, k=3, max_iters=100, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        np.random.seed(random_state)
    
    def fit(self, X):
        """
        Entraîne le modèle K-Means
        
        Parameters:
        X (array): Données à clusteriser
        
        Returns:
        array: Labels des clusters
        """
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iters):
            # Étape d'affectation
            clusters = self._create_clusters(X)
            
            # Vérification des clusters vides
            if self._has_empty_clusters(clusters):
                # Réinitialiser les centroïdes des clusters vides
                self._handle_empty_clusters(X, clusters)
            
            # Étape de mise à jour
            old_centroids = self.centroids.copy()
            self.centroids = self._update_centroids(X, clusters)
            
            # Vérification de la convergence
            if np.allclose(old_centroids, self.centroids):
                break
        
        # Retourne les labels
        return self._get_labels(X, clusters)
    
    def _initialize_centroids(self, X):
        """Initialise les centroïdes de manière aléatoire"""
        indices = np.random.choice(len(X), self.k, replace=False)
        return X[indices]
    
    def _create_clusters(self, X):
        """Crée les clusters en assignant chaque point au centroïde le plus proche"""
        clusters = [[] for _ in range(self.k)]
        
        for idx, point in enumerate(X):
            distances = np.sqrt(np.sum((point - self.centroids) ** 2, axis=1))
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(idx)
        
        return clusters
    
    def _has_empty_clusters(self, clusters):
        """Vérifie s'il y a des clusters vides"""
        return any(len(cluster) == 0 for cluster in clusters)
    
    def _handle_empty_clusters(self, X, clusters):
        """Gère les clusters vides en réinitialisant leurs centroïdes"""
        for i, cluster in enumerate(clusters):
            if len(cluster) == 0:
                # Choisir un point aléatoire comme nouveau centroïde
                random_point_idx = np.random.randint(len(X))
                self.centroids[i] = X[random_point_idx]
    
    def _update_centroids(self, X, clusters):
        """Met à jour les centroïdes"""
        centroids = np.zeros((self.k, X.shape[1]))
        
        for cluster_idx, cluster in enumerate(clusters):
            if cluster:  # Éviter la division par zéro
                cluster_mean = np.mean(X[cluster], axis=0)
                centroids[cluster_idx] = cluster_mean
        
        return centroids
    
    def _get_labels(self, X, clusters):
        """Convertit les clusters en labels"""
        labels = np.zeros(len(X))
        for cluster_idx, cluster in enumerate(clusters):
            labels[cluster] = cluster_idx
        return labels
    
    def predict(self, X):
        """
        Prédit les clusters pour de nouvelles données
        """
        labels = []
        for point in X:
            distances = np.sqrt(np.sum((point - self.centroids) ** 2, axis=1))
            cluster_idx = np.argmin(distances)
            labels.append(cluster_idx)
        return np.array(labels)