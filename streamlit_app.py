# streamlit_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, silhouette_score, classification_report
from sklearn.datasets import make_blobs, make_classification
import sys
import os

# Ajouter le chemin pour importer vos modules
sys.path.append(os.path.dirname(__file__))

# Importer vos algorithmes
from algorithms.knn import KNN
from algorithms.kmeans import KMeans
from algorithms.hierarchical import HierarchicalClustering

# Configuration de la page
st.set_page_config(
    page_title="ML Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS professionnel
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* Algorithm Cards */
    .algorithm-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        margin: 1.5rem 0;
        border-left: 4px solid #667eea;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .algorithm-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    .algorithm-card h2 {
        color: #2d3748;
        font-weight: 600;
        margin-top: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
        border-top: 3px solid #667eea;
    }
    
    .metric-container .metric-label {
        color: #718096;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-container .metric-value {
        color: #2d3748;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f0f0 0%, #f0f0f0 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
            
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #0ea5e9;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }
    
    /* SVG Icons */
    .icon {
        width: 24px;
        height: 24px;
        display: inline-block;
        vertical-align: middle;
    }
    
    /* DataFrames */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 8px;
        font-weight: 600;
        color: #2d3748;
    }
    
    /* Plot container */
    .plot-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration matplotlib pour un style professionnel
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class StreamlitDataGenerator:
    """Générateur de données adapté pour Streamlit"""
    
    @staticmethod
    def generate_classification_data(n_samples=300, noise=0.2, random_state=42):
        """Génère des données pour la classification"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            random_state=random_state,
            flip_y=noise
        )
        return X, y
    
    @staticmethod
    def generate_clustering_data(n_samples=300, n_clusters=4, cluster_std=0.8, random_state=42):
        """Génère des données pour le clustering"""
        X, y = make_blobs(
            n_samples=n_samples,
            centers=n_clusters,
            cluster_std=cluster_std,
            random_state=random_state
        )
        return X, y

def plot_comparison(X, y_true, y_pred, title):
    """Fonction de visualisation professionnelle pour Streamlit"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Style professionnel
    colors = sns.color_palette("husl", len(np.unique(y_true)))
    
    # Données réelles
    for i, label in enumerate(np.unique(y_true)):
        mask = y_true == label
        ax1.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], 
                   label=f'Class {label}', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax1.set_title('Ground Truth', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Feature 1', fontsize=11)
    ax1.set_ylabel('Feature 2', fontsize=11)
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Prédictions
    for i, label in enumerate(np.unique(y_pred)):
        mask = y_pred == label
        ax2.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], 
                   label=f'Cluster {label}', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax2.set_title('Predictions', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Feature 1', fontsize=11)
    ax2.set_ylabel('Feature 2', fontsize=11)
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    st.pyplot(fig)

def main():
    """Fonction principale de l'application Streamlit"""
    
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>Machine Learning Analytics Platform</h1>
        <p>Advanced algorithms exploration and performance comparison</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    
    algorithm = st.sidebar.radio(
        "Select Algorithm",
        ["KNN Classification", "K-Means Clustering", "Hierarchical Clustering", "Algorithm Comparison"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Info sidebar
    with st.sidebar.expander("About this Platform"):
        st.markdown("""
        This platform provides interactive exploration of fundamental machine learning algorithms:
        
        - **KNN**: Supervised classification
        - **K-Means**: Unsupervised clustering
        - **HAC**: Hierarchical clustering
        
        Adjust parameters in real-time and visualize results instantly.
        """)
    
    # Génération des données (commune à tous les algorithmes)
    data_gen = StreamlitDataGenerator()
    
    if algorithm == "KNN Classification":
        show_knn_demo(data_gen)
    
    elif algorithm == "K-Means Clustering":
        show_kmeans_demo(data_gen)
    
    elif algorithm == "Hierarchical Clustering":
        show_hierarchical_demo(data_gen)
    
    else:
        show_comparison(data_gen)

def show_knn_demo(data_gen):
    """Démonstration interactive de KNN"""
    
    st.markdown('<div class="algorithm-card">', unsafe_allow_html=True)
    st.markdown("## K-Nearest Neighbors (KNN)")
    st.markdown("""
    Supervised classification algorithm that predicts the class of a data point 
    based on the majority vote of its k nearest neighbors.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Paramètres KNN
    st.subheader("Configuration Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        k_value = st.slider("Number of neighbors (k)", 1, 15, 5, 
                           help="Number of nearest neighbors to consider for classification")
        test_size = st.slider("Test set size (%)", 10, 50, 30,
                             help="Percentage of data used for testing")
    
    with col2:
        n_samples = st.slider("Sample size", 100, 1000, 300, step=50,
                             help="Total number of samples to generate")
        noise_level = st.slider("Noise level", 0.0, 0.5, 0.1, step=0.05,
                               help="Amount of random noise in the data")
    
    with col3:
        random_state = st.number_input("Random seed", 0, 100, 42,
                                      help="Seed for reproducible results")
        regenerate = st.button("Generate New Data", use_container_width=True)
        if regenerate:
            random_state = np.random.randint(0, 100)
    
    # Génération des données
    X, y = data_gen.generate_classification_data(
        n_samples=n_samples, 
        noise=noise_level, 
        random_state=random_state
    )
    
    # Split des données
    split_idx = int(len(X) * (1 - test_size/100))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Entraînement et prédiction
    with st.spinner('Training model...'):
        knn = KNN(k=k_value)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
    # Affichage des résultats
    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{accuracy:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">K Value</div>
            <div class="metric-value">{k_value}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Training Samples</div>
            <div class="metric-value">{len(X_train)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Test Samples</div>
            <div class="metric-value">{len(X_test)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Visualisation
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Classification Results")
        plot_comparison(X_test, y_test, y_pred, f"KNN Classifier (k={k_value}) - Accuracy: {accuracy:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Rapport de classification
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='Blues').format("{:.3f}"), 
                    use_container_width=True)
    
    # Explication
    with st.expander("Algorithm Details"):
        st.markdown("""
        ### How KNN Works
        
        1. **Distance Calculation**: Compute distances between the test point and all training points
        2. **Neighbor Selection**: Select the k nearest neighbors
        3. **Majority Voting**: Determine the most common class among neighbors
        4. **Class Assignment**: Assign the winning class to the test point
        
        #### Advantages
        - Simple and intuitive algorithm
        - No training phase required
        - Naturally adapts to new data
        - Works well with multi-class problems
        
        #### Limitations
        - Computationally expensive for large datasets
        - Sensitive to the choice of k
        - Affected by irrelevant features
        - Requires feature scaling
        """)

def show_kmeans_demo(data_gen):
    """Démonstration interactive de K-Means"""
    
    st.markdown('<div class="algorithm-card">', unsafe_allow_html=True)
    st.markdown("## K-Means Clustering")
    st.markdown("""
    Unsupervised clustering algorithm that partitions data into k clusters 
    by minimizing within-cluster variance.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Paramètres K-Means
    st.subheader("Configuration Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        k_clusters = st.slider("Number of clusters (k)", 2, 10, 4)
        n_samples = st.slider("Sample size", 100, 1000, 300, step=50)
    
    with col2:
        cluster_std = st.slider("Cluster dispersion", 0.1, 2.0, 0.8, step=0.1)
        max_iters = st.slider("Maximum iterations", 10, 200, 100, step=10)
    
    with col3:
        random_state = st.number_input("Random seed", 0, 100, 42)
        regenerate = st.button("Generate New Data", use_container_width=True)
        if regenerate:
            random_state = np.random.randint(0, 100)
    
    # Génération des données
    X, y_true = data_gen.generate_clustering_data(
        n_samples=n_samples,
        n_clusters=k_clusters,
        cluster_std=cluster_std,
        random_state=random_state
    )
    
    # Clustering avec K-Means
    with st.spinner('Running K-Means algorithm...'):
        kmeans = KMeans(k=k_clusters, max_iters=max_iters, random_state=random_state)
        labels = kmeans.fit(X)
        silhouette = silhouette_score(X, labels)
    
    # Affichage des métriques
    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Silhouette Score</div>
            <div class="metric-value">{silhouette:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Clusters (k)</div>
            <div class="metric-value">{k_clusters}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        unique_clusters = len(np.unique(labels))
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Clusters Found</div>
            <div class="metric-value">{unique_clusters}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Avg Points/Cluster</div>
            <div class="metric-value">{len(X)//unique_clusters}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualisation
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("Clustering Results")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Vrais clusters
    colors_true = sns.color_palette("husl", k_clusters)
    for i in range(k_clusters):
        mask = y_true == i
        ax1.scatter(X[mask, 0], X[mask, 1], c=[colors_true[i]], 
                   label=f'Cluster {i}', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax1.set_title('Ground Truth Clusters', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Feature 1', fontsize=11)
    ax1.set_ylabel('Feature 2', fontsize=11)
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Clusters K-Means
    colors_pred = sns.color_palette("husl", k_clusters)
    for i in range(k_clusters):
        mask = labels == i
        ax2.scatter(X[mask, 0], X[mask, 1], c=[colors_pred[i]], 
                   label=f'Cluster {i}', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    
    # Centroïdes
    ax2.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                marker='X', s=300, c='red', edgecolors='black', linewidth=2,
                label='Centroids', zorder=5)
    ax2.set_title(f'K-Means Clusters (Silhouette: {silhouette:.3f})', 
                 fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Feature 1', fontsize=11)
    ax2.set_ylabel('Feature 2', fontsize=11)
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Méthode du coude
    st.subheader("Elbow Method for Optimal k")
    
    if st.button("Calculate Elbow Method", use_container_width=True):
        with st.spinner("Computing inertia for different k values..."):
            inertias = []
            silhouettes = []
            k_range = range(2, 11)
            
            progress_bar = st.progress(0)
            
            for idx, k in enumerate(k_range):
                kmeans_temp = KMeans(k=k, max_iters=100, random_state=42)
                temp_labels = kmeans_temp.fit(X)
                
                # Calcul de l'inertie
                inertia = 0
                for i in range(len(X)):
                    print(f"temp_labels[i]: {temp_labels[i]}, type: {type(temp_labels[i])}")

                    centroid = kmeans_temp.centroids[temp_labels[i]]
                    inertia += np.sum((X[i] - centroid) ** 2)
                inertias.append(inertia)
                
                # Silhouette score
                silhouettes.append(silhouette_score(X, temp_labels))
                
                progress_bar.progress((idx + 1) / len(k_range))
            
            progress_bar.empty()
            
            # Graphiques
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Elbow plot
            ax1.plot(k_range, inertias, 'o-', linewidth=2, markersize=8, color='#667eea')
            ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
            ax1.set_ylabel('Inertia (Within-cluster variance)', fontsize=11)
            ax1.set_title('Elbow Method', fontsize=14, fontweight='bold', pad=15)
            ax1.grid(True, alpha=0.3)
            
            # Silhouette plot
            ax2.plot(k_range, silhouettes, 'o-', linewidth=2, markersize=8, color='#10b981')
            ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
            ax2.set_ylabel('Silhouette Score', fontsize=11)
            ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold', pad=15)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Explication
    with st.expander("Algorithm Details"):
        st.markdown("""
        ### How K-Means Works
        
        1. **Initialization**: Randomly select k centroids
        2. **Assignment**: Assign each point to the nearest centroid
        3. **Update**: Recalculate centroids as the mean of assigned points
        4. **Iterate**: Repeat steps 2-3 until convergence
        
        #### Advantages
        - Simple and efficient algorithm
        - Scales well to large datasets
        - Guaranteed to converge
        - Works well with spherical clusters
        
        #### Limitations
        - Requires specifying k in advance
        - Sensitive to initial centroid placement
        - Assumes spherical clusters of similar size
        - Affected by outliers
        """)

def show_hierarchical_demo(data_gen):
    """Démonstration interactive de la CAH"""
    
    st.markdown('<div class="algorithm-card">', unsafe_allow_html=True)
    st.markdown("## Hierarchical Agglomerative Clustering (HAC)")
    st.markdown("""
    Hierarchical clustering algorithm that builds a tree of clusters 
    by iteratively merging the closest groups.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Paramètres CAH
    st.subheader("Configuration Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_samples = st.slider("Sample size", 50, 300, 100, step=25,
                             help="Limited for computational efficiency")
        n_clusters_true = st.slider("True clusters", 2, 6, 3)
    
    with col2:
        cluster_std = st.slider("Cluster dispersion", 0.3, 1.5, 0.6, step=0.1)
        final_clusters = st.slider("Final clusters", 2, 10, 3)
    
    with col3:
        method = st.selectbox("Linkage method", ["ward", "complete", "average", "single"],
                             help="Method for calculating cluster distances")
        random_state = st.number_input("Random seed", 0, 100, 42)
        regenerate = st.button("Generate New Data", use_container_width=True)
        if regenerate:
            random_state = np.random.randint(0, 100)
    
    # Génération des données
    X, y_true = data_gen.generate_clustering_data(
        n_samples=n_samples,
        n_clusters=n_clusters_true,
        cluster_std=cluster_std,
        random_state=random_state
    )
    
    # Clustering hiérarchique
    with st.spinner('Building hierarchical clusters...'):
        cah = HierarchicalClustering(method=method)
        Z = cah.fit(X)
        labels = cah.get_clusters(final_clusters)
        silhouette = silhouette_score(X, labels)
    
    # Affichage des métriques
    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Silhouette Score</div>
            <div class="metric-value">{silhouette:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Linkage Method</div>
            <div class="metric-value" style="font-size: 1.3rem;">{method.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Selected Clusters</div>
            <div class="metric-value">{final_clusters}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Sample Size</div>
            <div class="metric-value">{n_samples}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualisations
    st.subheader("Clustering Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Dendrogramme
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown("**Dendrogram**")
        fig_dendo, ax = plt.subplots(figsize=(10, 6))
        from scipy.cluster.hierarchy import dendrogram
        dendrogram(Z, ax=ax, color_threshold=0, above_threshold_color='#667eea')
        
        cut_height = st.slider("Cut height", 0.0, float(Z[:, 2].max()), 
                              float(Z[:, 2].max()) * 0.5, key="cut_height")
        
        ax.axhline(y=cut_height, color='red', linestyle='--', linewidth=2, label='Cut threshold')
        ax.set_title(f'Hierarchical Clustering Dendrogram ({method})', 
                    fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel('Distance', fontsize=10)
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig_dendo)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Clusters résultants
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.markdown("**HAC Clusters**")
        fig_clusters, ax = plt.subplots(figsize=(10, 6))
        
        colors = sns.color_palette("husl", final_clusters)
        for i in range(final_clusters):
            mask = labels == i
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], 
                      label=f'Cluster {i}', alpha=0.7, s=60, 
                      edgecolors='white', linewidth=0.5)
        
        ax.set_title(f'HAC Results - {final_clusters} Clusters (Silhouette: {silhouette:.3f})', 
                    fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('Feature 1', fontsize=10)
        ax.set_ylabel('Feature 2', fontsize=10)
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_clusters)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Comparaison avec les vrais clusters
    st.subheader("Ground Truth Comparison")
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    
    fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Vrais clusters
    colors_true = sns.color_palette("husl", n_clusters_true)
    for i in range(n_clusters_true):
        mask = y_true == i
        ax1.scatter(X[mask, 0], X[mask, 1], c=[colors_true[i]], 
                   label=f'Cluster {i}', alpha=0.7, s=60, 
                   edgecolors='white', linewidth=0.5)
    ax1.set_title('Ground Truth Clusters', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Feature 1', fontsize=11)
    ax1.set_ylabel('Feature 2', fontsize=11)
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Clusters CAH
    colors_pred = sns.color_palette("husl", final_clusters)
    for i in range(final_clusters):
        mask = labels == i
        ax2.scatter(X[mask, 0], X[mask, 1], c=[colors_pred[i]], 
                   label=f'Cluster {i}', alpha=0.7, s=60, 
                   edgecolors='white', linewidth=0.5)
    ax2.set_title(f'HAC Clusters ({method})', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Feature 1', fontsize=11)
    ax2.set_ylabel('Feature 2', fontsize=11)
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_comp)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Explication
    with st.expander("Algorithm Details"):
        st.markdown("""
        ### How Hierarchical Clustering Works
        
        1. **Initialization**: Each data point starts as its own cluster
        2. **Find Closest Pair**: Identify the two closest clusters
        3. **Merge**: Combine these clusters into one
        4. **Repeat**: Continue until only one cluster remains
        
        #### Linkage Methods
        
        - **Single**: Minimum distance between clusters
        - **Complete**: Maximum distance between clusters
        - **Average**: Average distance between all pairs
        - **Ward**: Minimizes within-cluster variance
        
        #### Advantages
        - Hierarchical visualization (dendrogram)
        - No need to specify k in advance
        - Captures complex cluster structures
        - Deterministic results
        
        #### Limitations
        - High computational complexity O(n³)
        - Sensitive to noise and outliers
        - Difficult to determine optimal cut height
        - Not suitable for large datasets
        """)

def show_comparison(data_gen):
    """Page de comparaison professionnelle des algorithmes"""
    
    st.markdown('<div class="algorithm-card">', unsafe_allow_html=True)
    st.markdown("## Algorithm Comparison & Benchmarking")
    st.markdown("""
    Comprehensive performance analysis and comparison of KNN, K-Means, and HAC 
    algorithms on identical datasets.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Paramètres communs
    st.subheader("Dataset Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        n_samples = st.slider("Sample size", 200, 500, 300, step=50)
        n_clusters = st.slider("Number of clusters", 2, 5, 3)
    
    with col2:
        cluster_std = st.slider("Cluster dispersion", 0.3, 1.2, 0.7, step=0.1)
        k_value = st.slider("K for KNN", 1, 10, 5)
    
    with col3:
        test_split = st.slider("Test split %", 20, 40, 30)
        max_iters = st.slider("Max iterations", 50, 200, 100, step=25)
    
    with col4:
        random_state = st.number_input("Random seed", 0, 100, 42)
        regenerate = st.button("Run New Experiment", use_container_width=True)
        if regenerate:
            random_state = np.random.randint(0, 100)
    
    # Génération des données
    with st.spinner('Generating dataset...'):
        X, y_true = data_gen.generate_clustering_data(
            n_samples=n_samples,
            n_clusters=n_clusters,
            cluster_std=cluster_std,
            random_state=random_state
        )
    
    # Application des trois algorithmes
    results = {}
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # KNN
    progress_text.text("Running KNN Classification...")
    split_idx = int(len(X) * (1 - test_split/100))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_true[:split_idx], y_true[split_idx:]
    
    knn = KNN(k=k_value)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    results['KNN'] = {
        'labels': y_pred_knn,
        'accuracy': accuracy_score(y_test, y_pred_knn),
        'data': X_test,
        'y_true': y_test
    }
    progress_bar.progress(33)
    
    # K-Means
    progress_text.text("Running K-Means Clustering...")
    kmeans = KMeans(k=n_clusters, max_iters=max_iters, random_state=random_state)
    labels_kmeans = kmeans.fit(X)
    results['K-Means'] = {
        'labels': labels_kmeans,
        'silhouette': silhouette_score(X, labels_kmeans),
        'data': X,
        'centroids': kmeans.centroids
    }
    progress_bar.progress(66)
    
    # CAH
    progress_text.text("Running Hierarchical Clustering...")
    cah = HierarchicalClustering(method='ward')
    Z = cah.fit(X)
    labels_cah = cah.get_clusters(n_clusters)
    results['HAC'] = {
        'labels': labels_cah,
        'silhouette': silhouette_score(X, labels_cah),
        'data': X
    }
    progress_bar.progress(100)
    
    progress_text.empty()
    progress_bar.empty()
    
    # Affichage des métriques comparatives
    st.subheader("Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container" style="border-top-color: #3b82f6;">
            <div class="metric-label">KNN Accuracy</div>
            <div class="metric-value" style="color: #3b82f6;">{results['KNN']['accuracy']:.3f}</div>
            <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #64748b;">
                Classification Score
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container" style="border-top-color: #10b981;">
            <div class="metric-label">K-Means Silhouette</div>
            <div class="metric-value" style="color: #10b981;">{results['K-Means']['silhouette']:.3f}</div>
            <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #64748b;">
                Cluster Quality
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container" style="border-top-color: #f59e0b;">
            <div class="metric-label">HAC Silhouette</div>
            <div class="metric-value" style="color: #f59e0b;">{results['HAC']['silhouette']:.3f}</div>
            <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #64748b;">
                Cluster Quality
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualisation comparative
    st.subheader("Visual Comparison")
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    colors = sns.color_palette("husl", n_clusters)
    
    # Données originales
    for i in range(n_clusters):
        mask = y_true == i
        axes[0, 0].scatter(X[mask, 0], X[mask, 1], c=[colors[i]], 
                          label=f'Cluster {i}', alpha=0.7, s=60, 
                          edgecolors='white', linewidth=0.5)
    axes[0, 0].set_title('Ground Truth Data', fontsize=14, fontweight='bold', pad=15)
    axes[0, 0].set_xlabel('Feature 1', fontsize=11)
    axes[0, 0].set_ylabel('Feature 2', fontsize=11)
    axes[0, 0].legend(frameon=True, shadow=True, loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    
    # KNN
    unique_knn = np.unique(results['KNN']['labels'])
    colors_knn = sns.color_palette("husl", len(unique_knn))
    for idx, i in enumerate(unique_knn):
        mask = results['KNN']['labels'] == i
        axes[0, 1].scatter(results['KNN']['data'][mask, 0], 
                          results['KNN']['data'][mask, 1], 
                          c=[colors_knn[idx]], label=f'Class {i}', 
                          alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    axes[0, 1].set_title(f'KNN Classification (Accuracy: {results["KNN"]["accuracy"]:.3f})', 
                        fontsize=14, fontweight='bold', pad=15)
    axes[0, 1].set_xlabel('Feature 1', fontsize=11)
    axes[0, 1].set_ylabel('Feature 2', fontsize=11)
    axes[0, 1].legend(frameon=True, shadow=True, loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    
    # K-Means
    for i in range(n_clusters):
        mask = results['K-Means']['labels'] == i
        axes[1, 0].scatter(results['K-Means']['data'][mask, 0], 
                          results['K-Means']['data'][mask, 1], 
                          c=[colors[i]], label=f'Cluster {i}', 
                          alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    axes[1, 0].scatter(results['K-Means']['centroids'][:, 0], 
                      results['K-Means']['centroids'][:, 1], 
                      marker='X', s=300, c='red', edgecolors='black', 
                      linewidth=2, label='Centroids', zorder=5)
    axes[1, 0].set_title(f'K-Means Clustering (Silhouette: {results["K-Means"]["silhouette"]:.3f})', 
                        fontsize=14, fontweight='bold', pad=15)
    axes[1, 0].set_xlabel('Feature 1', fontsize=11)
    axes[1, 0].set_ylabel('Feature 2', fontsize=11)
    axes[1, 0].legend(frameon=True, shadow=True, loc='best')
    axes[1, 0].grid(True, alpha=0.3)
    
    # HAC
    for i in range(n_clusters):
        mask = results['HAC']['labels'] == i
        axes[1, 1].scatter(results['HAC']['data'][mask, 0], 
                          results['HAC']['data'][mask, 1], 
                          c=[colors[i]], label=f'Cluster {i}', 
                          alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    axes[1, 1].set_title(f'HAC Clustering (Silhouette: {results["HAC"]["silhouette"]:.3f})', 
                        fontsize=14, fontweight='bold', pad=15)
    axes[1, 1].set_xlabel('Feature 1', fontsize=11)
    axes[1, 1].set_ylabel('Feature 2', fontsize=11)
    axes[1, 1].legend(frameon=True, shadow=True, loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tableau comparatif détaillé
    st.subheader("Detailed Algorithm Comparison")
    
    comparison_data = {
        'Algorithm': ['KNN', 'K-Means', 'HAC'],
        'Type': ['Supervised', 'Unsupervised', 'Unsupervised'],
        'Time Complexity': ['O(nd)', 'O(ndk)', 'O(n³)'],
        'Space Complexity': ['O(n)', 'O(nk)', 'O(n²)'],
        'Performance Score': [
            f"{results['KNN']['accuracy']:.4f}",
            f"{results['K-Means']['silhouette']:.4f}", 
            f"{results['HAC']['silhouette']:.4f}"
        ],
        'Scalability': ['Medium', 'High', 'Low'],
        'Best Use Case': [
            'Labeled classification tasks',
            'Large-scale clustering',
            'Exploratory data analysis'
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    st.dataframe(
        df_comparison.style.set_properties(**{
            'background-color': 'white',
            'color': '#2d3748',
            'border-color': '#e2e8f0'
        }).set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#667eea'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center')
            ]}
        ]),
        use_container_width=True,
        height=250
    )
    
    # Graphique radar de comparaison
    st.subheader("Multi-Criteria Evaluation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        
        # Critères normalisés (1-10)
        criteria = {
            'KNN': {
                'Speed': 6,
                'Accuracy': results['KNN']['accuracy'] * 10,
                'Scalability': 5,
                'Simplicity': 9,
                'Interpretability': 8
            },
            'K-Means': {
                'Speed': 9,
                'Accuracy': results['K-Means']['silhouette'] * 10,
                'Scalability': 9,
                'Simplicity': 8,
                'Interpretability': 7
            },
            'HAC': {
                'Speed': 3,
                'Accuracy': results['HAC']['silhouette'] * 10,
                'Scalability': 3,
                'Simplicity': 6,
                'Interpretability': 9
            }
        }
        
        categories = list(criteria['KNN'].keys())
        fig_radar = plt.figure(figsize=(10, 8))
        ax = fig_radar.add_subplot(111, projection='polar')
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        colors_algo = ['#3b82f6', '#10b981', '#f59e0b']
        algo_names = ['KNN', 'K-Means', 'HAC']
        
        for idx, algo in enumerate(algo_names):
            values = list(criteria[algo].values())
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=algo, color=colors_algo[idx])
            ax.fill(angles, values, alpha=0.15, color=colors_algo[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], size=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, shadow=True)
        plt.title('Algorithm Performance Radar', fontsize=14, fontweight='bold', pad=20)
        
        st.pyplot(fig_radar)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Selection Guide")
        
        st.markdown("""
        <div class="info-box">
            <strong>KNN</strong><br>
            ✓ Classification problems<br>
            ✓ Small to medium datasets<br>
            ✓ Need for simplicity<br>
            ✓ Labeled data available
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
            <strong>K-Means</strong><br>
            ✓ Unsupervised clustering<br>
            ✓ Large datasets<br>
            ✓ Spherical clusters<br>
            ✓ Speed is critical
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <strong>HAC</strong><br>
            ✓ Exploratory analysis<br>
            ✓ Hierarchical structure<br>
            ✓ Visualization needed<br>
            ✓ Small datasets
        </div>
        """, unsafe_allow_html=True)
    
    # Recommandation intelligente
    st.subheader("Smart Recommendation")
    
    best_algo = max(
        [('KNN', results['KNN']['accuracy']), 
         ('K-Means', results['K-Means']['silhouette']), 
         ('HAC', results['HAC']['silhouette'])],
        key=lambda x: x[1]
    )
    
    st.success(f"""
    **Recommended Algorithm: {best_algo[0]}**
    
    Based on the current dataset characteristics and performance metrics, 
    {best_algo[0]} achieved the best score of {best_algo[1]:.4f}.
    
    This recommendation considers the dataset size ({n_samples} samples), 
    cluster structure ({n_clusters} clusters), and computational requirements.
    """)

if __name__ == "__main__":
    main()