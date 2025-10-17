import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithms.knn import KNN
from utils.data_loader import DataGenerator
from utils.visualizer import Visualizer
from utils.evaluator import Evaluator

def run_knn_example():
    """Exemple complet d'utilisation de KNN"""
    print("=== EXEMPLE KNN ===")
    
    # Génération des données
    X_train, X_test, y_train, y_test = DataGenerator.generate_classification_data()
    
    # Initialisation et entraînement
    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    
    # Prédiction et évaluation
    y_pred = knn.predict(X_test)
    accuracy = Evaluator.evaluate_classification(y_test, y_pred, "KNN")
    
    # Visualisation
    viz = Visualizer()
    viz.plot_classification_results(X_test, y_test, y_pred, "KNN - Résultats")
    viz.plot_confusion_matrix(y_test, y_pred, "KNN - Matrice de Confusion")
    
    return knn, accuracy

if __name__ == "__main__":
    run_knn_example()