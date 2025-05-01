import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps as cm
import csv
import numpy as np
import statistics
import networkx as nx
import random
import copy
import collections
import math
import scipy as sp
import torch
from sklearn.cluster import KMeans, SpectralClustering
from cdlib import algorithms as algo

def create_undirected_graph_from_edge_list(file_path):
    G = nx.Graph()  
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            node1, node2 = line.strip().split() 
            G.add_edge(node1, node2)
    return G

def select_connected_nodes(graph, num_nodes):

    if len(graph) < num_nodes:
        return None

    start_node = random.choice(list(graph.nodes))
    bfs_nodes = list(nx.bfs_tree(graph, start_node).nodes)
    selected_nodes = bfs_nodes[:min(num_nodes, len(bfs_nodes))]

    return selected_nodes


def compute_laplacian(adj_matrix, normalization='sym'):

    degree = torch.sum(adj_matrix, dim=-1)
    degree_matrix = torch.diag(degree)
    laplacian_matrix = degree_matrix - adj_matrix

    if normalization is None:
        return laplacian_matrix

    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.
    degree_inv_sqrt_matrix = torch.diag(degree_inv_sqrt)

    if normalization == 'sym':
        return torch.mm(degree_inv_sqrt_matrix, torch.mm(laplacian_matrix, degree_inv_sqrt_matrix))
    elif normalization == 'rw':
      degree_inv = torch.pow(degree, -1)
      degree_inv[torch.isinf(degree_inv)] = 0.
      degree_inv_matrix = torch.diag(degree_inv)
      return torch.mm(degree_inv_matrix, laplacian_matrix)
    else:
        raise ValueError("Invalid normalization method.")

def spectral_clustering(adj_matrix, k, normalization='sym'):
    """Performs spectral clustering on an adjacency matrix.

    Args:
        adj_matrix (torch.Tensor): Adjacency matrix.
        k (int): Number of clusters.
        normalization (str, optional): Normalization method for Laplacian. 
            'sym' for symmetric normalized, 'rw' for random walk, 
            None for unnormalized. Defaults to 'sym'.

    Returns:
        numpy.ndarray: Cluster labels for each node.
    """
    laplacian_matrix = compute_laplacian(adj_matrix, normalization)
    eigenvalues, eigenvectors = torch.linalg.eig(laplacian_matrix)
    eigenvalues = eigenvalues.real
    sorted_indices = torch.argsort(eigenvalues)
    selected_eigenvectors = eigenvectors[:, sorted_indices[:k]].real
    normalized_eigenvectors = selected_eigenvectors / torch.linalg.norm(selected_eigenvectors, dim=1, keepdim=True)
    kmeans = KMeans(n_clusters=k, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized_eigenvectors)
    return cluster_labels

def labels_to_sets(labels, nodelist):
    cluster_sets = {}
    i = 0
    for node in nodelist:
        cluster_id = labels[i]
        if cluster_id is not None:
            if cluster_id not in cluster_sets:
                cluster_sets[cluster_id] = set()
            cluster_sets[cluster_id].add(node)

        i += 1
    return list(cluster_sets.values())

def perform_spectral_clustering_2(graph, n_clusters):
    adj_matrix = nx.to_numpy_array(graph)
    spectral_clustering = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='precomputed', 
        assign_labels='kmeans',
        random_state=0
    )
    clusters = spectral_clustering.fit_predict(adj_matrix)
    return clusters
    

def compute_modularity(graph, partition):
    modularity = nx.community.modularity(graph, partition)
    return modularity


def save_to_csv(filename, data, headers=None):
    with open(filename, 'w', newline='') as csvfile:
        if headers:
            writer = csv.writer(csvfile)
            writer.writerows([headers])  # Write headers
            writer.writerows(data)
        elif data and all(isinstance(row, list) for row in data):
             writer = csv.writer(csvfile)
             writer.writerows(data)
        else:
            raise ValueError("Data must be a list of lists or a list of dictionaries when headers are not provided.")



if __name__ == "__main__":


    if (device := torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        print(f"Using device: {device}")


    file_path = "roadNet-CA.txt"
    graph = create_undirected_graph_from_edge_list(file_path)

    cc_generator = nx.connected_components(graph)
    lcc = max(cc_generator, key=len)
    lcc = graph.subgraph(lcc)

    list_of_lists = []


    for i in range(10):
    
        connected_nodes = select_connected_nodes(lcc, 10000)
        subgraph = lcc.subgraph(connected_nodes)
        nodeList = list(subgraph.nodes())
        
        comms_louvain = nx.community.louvain_communities(subgraph)
        modularity_louvain = compute_modularity(subgraph, comms_louvain)

        print("Louvain Successfully computed")

        k = len(comms_louvain)
        adjacency_matrix = nx.to_numpy_array(subgraph)
        adj_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        cluster_labels = spectral_clustering(adj_matrix, k)
        cluster_sets = labels_to_sets(cluster_labels, nodeList)
        modularity_spectralClustering = compute_modularity(subgraph, cluster_sets)

        print("Spectral Clustering Successfully computed")

        cluster_labels_2 = perform_spectral_clustering_2(subgraph, k)
        cluster_sets_2 = labels_to_sets(cluster_labels_2, nodeList)
        modularity_spectralClustering_2 = compute_modularity(subgraph, cluster_sets_2)

        print("Spectral Clustering 2 Successfully computed")

        list_of_modularities = [modularity_louvain, modularity_spectralClustering, modularity_spectralClustering_2]
        list_of_lists.append(list_of_modularities)


        file = open("spectralClusteringOutput.txt", "a", encoding="utf-8")
        text = "Louvain Modularity: " + str(modularity_louvain) + "; " + "Spectral Clustering Modularity: " + str(modularity_spectralClustering) + "; " + "Spectral Clustering 2 Modularity: " + str(modularity_spectralClustering_2) + "\n"
        file.write(text)
        file.close()
        print("Done with iteration", i + 1, "\n")

    headers = ["Louvain Modularity", "Spectral Clustering Modularity", "Spectral Clustering 2 Modularity"]
    save_to_csv("modularity_results.csv", list_of_lists, headers)
    print("Results saved to modularity_results.csv")

    df_results = pd.read_csv('modularity_scores.csv')
    df_results.columns = ['Methods', 'Scores']


    sns.swarmplot(data=df_results, x='Methods', y='Scores', palette='Set2')
    plt.title('Modularity Scores for Different Methods')
    plt.xlabel('Methods')
    plt.ylabel('Modularity Scores')
    plt.xticks(rotation=7)
    plt.savefig('modularity_scores_plot.png')
    plt.show()
    
