import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

class Evaluator:
    """Classe pour évaluer les performances des modèles"""
    
    @staticmethod
    def evaluate_classification(y_true, y_pred, model_name="Modèle"):
        """
        Évalue un modèle de classification
        """
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"=== ÉVALUATION {model_name.upper()} ===")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nRapport de classification:")
        print(classification_report(y_true, y_pred))
        
        return accuracy
    
    @staticmethod
    def evaluate_clustering(X, labels_true, labels_pred, model_name="Modèle"):
        """
        Évalue un modèle de clustering
        """
        silhouette = silhouette_score(X, labels_pred)
        ari = adjusted_rand_score(labels_true, labels_pred)
        nmi = normalized_mutual_info_score(labels_true, labels_pred)
        
        print(f"=== ÉVALUATION {model_name.upper()} ===")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Adjusted Rand Index: {ari:.4f}")
        print(f"Normalized Mutual Info: {nmi:.4f}")
        
        return {
            'silhouette': silhouette,
            'ari': ari,
            'nmi': nmi
        }
    
    @staticmethod
    def compare_algorithms():
        """
        Affiche une comparaison des algorithmes
        """
        comparison = {
            'Algorithme': ['KNN', 'K-Means', 'CAH'],
            'Type': ['Supervisé', 'Non-supervisé', 'Non-supervisé'],
            'Complexité': ['O(nd)', 'O(ndk)', 'O(n³)'],
            'Avantages': [
                'Simple, pas d\'entraînement',
                'Évolutif, facile à implémenter',
                'Hiérarchique, visualisation'
            ],
            'Inconvénients': [
                'Sensible au bruit, lent en prédiction',
                'Sensible à l\'initialisation',
                'Coûteux computationnellement'
            ]
        }
        
        return comparison