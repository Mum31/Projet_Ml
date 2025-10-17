import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

class Visualizer:
    """Classe pour visualiser les résultats des algorithmes"""
    
    def __init__(self, figsize=(10, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
    
    def plot_classification_results(self, X, y_true, y_pred, title="Résultats de Classification"):
        """
        Visualise les résultats d'un modèle de classification
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Données réelles
        ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.8)
        ax1.set_title('Données Réelles')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        
        # Prédictions
        ax2.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.8)
        ax2.set_title('Prédictions')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def plot_clustering_results(self, X, labels, centroids=None, title="Résultats de Clustering"):
        """
        Visualise les résultats d'un algorithme de clustering
        """
        plt.figure(figsize=self.figsize)
        
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=50, alpha=0.7)
        
        if centroids is not None:
            plt.scatter(centroids[:, 0], centroids[:, 1], 
                       c='red', marker='X', s=200, label='Centroïdes')
        
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        if centroids is not None:
            plt.legend()
        plt.show()
    
    def plot_comparison(self, X, y_list, titles, figsize=(15, 5)):
        """
        Compare plusieurs résultats côte à côte
        """
        n_plots = len(y_list)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        if n_plots == 1:
            axes = [axes]
        
        for ax, y, title in zip(axes, y_list, titles):
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
            ax.set_title(title)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            plt.colorbar(scatter, ax=ax)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Matrice de Confusion"):
        """
        Affiche la matrice de confusion
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('Vrais labels')
        plt.xlabel('Labels prédits')
        plt.show()