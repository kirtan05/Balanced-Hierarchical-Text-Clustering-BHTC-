import csv
import json
import os
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

def aggregate_embeddings(cluster, embeddings_dict):
    """
    Aggregates embeddings within a cluster by averaging them.
    """
    cluster_embeddings = np.array([embeddings_dict[key] for key in cluster])
    return np.mean(cluster_embeddings, axis=0)

def constrained_clustering_general(embeddings_dict, cluster_size, total_entries):
    """
    Perform constrained clustering using hierarchical agglomerative clustering.
    """
    primary_keys = list(embeddings_dict.keys())
    embeddings = np.array(list(embeddings_dict.values()))
    
    Z = linkage(pdist(embeddings, metric='cosine'), method='average')
    
    n_clusters = total_entries // cluster_size
    initial_clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    cluster_dict = {}
    for i, cluster in enumerate(initial_clusters):
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append(primary_keys[i])

    exact_clusters = []
    overflow = []
    
    for cluster, keys in cluster_dict.items():
        if len(keys) == cluster_size:
            exact_clusters.append(keys)
        elif len(keys) < cluster_size:
            overflow.extend(keys)
        else:
            for i in range(0, len(keys), cluster_size):
                if i + cluster_size <= len(keys):
                    exact_clusters.append(keys[i:i + cluster_size])
                else:
                    overflow.extend(keys[i:])
    
    for i in range(0, len(overflow), cluster_size):
        exact_clusters.append(overflow[i:i + cluster_size])

    return exact_clusters, initial_clusters

def hierarchical_clustering(embeddings_dict, cluster_sizes):
    """
    Perform hierarchical clustering and export the results as a JSON Dump String.
    Parameters:
    - embeddings_dict: Dictionary of primary keys to embeddings.
    - cluster_sizes: List specifying the size of clusters at each level.
    """
    n_students = len(embeddings_dict)
    hierarchy = []
    json_output = {}
    for level, cluster_size in enumerate(cluster_sizes):
        print(f"Level {level+1}: Clustering into groups of size {cluster_size}")
        
        clusters, _ = constrained_clustering_general(embeddings_dict, cluster_size, n_students)
        
        hierarchy.append(clusters)
        json_output[f"Level {level+1}"] = []
        for cluster_id, cluster in enumerate(clusters, start=1):
            json_output[f"Level {level+1}"].append({
                "Cluster_ID": cluster_id,
                "Primary_Keys": cluster
            })
        new_embeddings_dict = {tuple(cluster): aggregate_embeddings(cluster, embeddings_dict) 
                               for cluster in clusters}
        embeddings_dict = new_embeddings_dict
        n_students = len(embeddings_dict)
    final_json = json.dumps(json_output,  indent=4)
    return final_json