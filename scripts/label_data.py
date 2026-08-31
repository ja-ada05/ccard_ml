from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
import pandas as pd

def umap_reduce(X, min_dist=0.1, n_neighbors=15, n_components=2, random_state=42):
    """
    Reduce dimensionality of data using UMAP.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data to reduce.
    min_dist : float, default=0.1
        Minimum distance between points in the low-dimensional embedding.
    n_neighbors : int, default=15
        Number of local neighbors used to build the manifold structure.
    n_components : int, default=2
        Dimension of the reduced space.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    X_embedded : ndarray of shape (n_samples, n_components)
        The reduced-dimensional representation.
    reducer : umap.UMAP
        The fitted UMAP object (for transforming new data later).
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )
    X_embedded = reducer.fit_transform(X)
    return X_embedded, reducer

def dim_reduc_umap(df:pd.DataFrame): 
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    reduced_df = umap_reduce(scaled_df,  n_components=8)
    reduced_df = pd.DataFrame(reduced_df[0])
    return reduced_df

def kmeans_label(reduced_df: pd.DataFrame, df:pd.DataFrame):
    kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
    labels = kmeans.fit_predict(reduced_df.copy())

    df['kmeans_labels'] = labels
    return df

