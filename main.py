import sys
import os

# Ajouter le chemin pour importer les modules
sys.path.append(os.path.join(os.path.dirname(__file__)))

from examples.knn_example import run_knn_example
from examples.kmeans_example import run_kmeans_example
from examples.hierarchical_example import run_hierarchical_example
from utils.evaluator import Evaluator
import pandas as pd

def main():
    """Fonction principale exécutant tous les exemples"""
    print("=" * 60)
    print("PROJET: KNN, K-MEANS ET CLASSIFICATION ASCENDANTE HIÉRARCHIQUE")
    print("=" * 60)
    
    # Exécution des exemples
    print("\n1. EXÉCUTION DES ALGORITHMES")
    print("-" * 40)
    
    # KNN
    knn_model, knn_accuracy = run_knn_example()
    
    # K-Means
    kmeans_model, kmeans_metrics = run_kmeans_example()
    
    # CAH
    cah_model, cah_metrics = run_hierarchical_example()
    
    # Comparaison finale
    print("\n2. COMPARAISON DES ALGORITHMES")
    print("-" * 40)
    
    comparison = Evaluator.compare_algorithms()
    df_comparison = pd.DataFrame(comparison)
    print("\nTableau comparatif:")
    print(df_comparison.to_string(index=False))
    
    # Applications industrielles
    print("\n3. APPLICATIONS INDUSTRIELLES")
    print("-" * 40)
    
    applications = {
        'KNN': [
            '• Reconnaissance de caractères',
            '• Systèmes de recommandation',
            '• Diagnostic médical'
        ],
        'K-Means': [
            '• Segmentation client',
            '• Compression d\'images',
            '• Détection d\'anomalies'
        ],
        'CAH': [
            '• Biologie (phylogénie)',
            '• Recherche documentaire',
            '• Analyse de marché'
        ]
    }
    
    for algo, apps in applications.items():
        print(f"\n{algo}:")
        for app in apps:
            print(f"  {app}")
    
    print("\n" + "=" * 60)
    print("PROJET TERMINÉ AVEC SUCCÈS!")
    print("=" * 60)

if __name__ == "__main__":
    main()