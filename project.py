import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps as cm
import csv
import numpy as np
import statistics
import networkx as nx
import osmnx as ox
import random
import powerlaw
import scipy.stats as stats
import copy
import community as community_louvain
import cdlib as cd
import collections
import networkit as nk
import pwlf
from scipy.stats import logser
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from netgraph import Graph
import math
from scipy.spatial import ConvexHull


NUMITERS = 30

def create_graph_from_edge_list(file_path):
    G = nx.DiGraph()  
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            node1, node2 = line.strip().split() 
            G.add_edge(node1, node2)
    return G


def create_undirected_graph_from_edge_list(file_path):
    G = nx.Graph()  
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            node1, node2 = line.strip().split() 
            G.add_edge(node1, node2)
    return G

def entropy(dist):
    """Returns the entropy of `dist` in bits (base-2)."""
    dist = np.asarray(dist)
    ent = np.nansum(dist * np.log2(1 / dist))
    return ent

def write_nodes_number_and_shortest_paths(graph, n_samples=1000000, output_path='shortest_path_length.txt'):

    print("Calculating the average shortest path length of the largest connected component of the California road network (with approximation) ...")

    with open(output_path, encoding='utf-8', mode='a') as f:
        records_length = []
        records_count = []

        for i in range(NUMITERS):
            lengths = 0
            count = 0
           
            all_nodes = list(graph.nodes)
            selected_nodes = random.sample(all_nodes, n_samples)
            subgraph1 = graph.subgraph(selected_nodes)
            cc_generator = nx.connected_components(subgraph1)
            for component in cc_generator:
                if len(component) > 1:
                    subgraph2 = subgraph1.subgraph(component)
                    distances = nx.johnson(subgraph2)
                    for key ,val in distances.items():
                        lengths += len(val)
                        count += 1

            records_length.append(lengths)
            records_count.append(count)

            f.write(f'The average shortest path length of the largest connected component of the California road network (with approximation) for the {i+1} iteration is: {lengths / count} \n')
            print(f'done {i+1} of {NUMITERS} iterations')

        avg = sum(records_length) / sum(records_count)
        print(f'The average shortest path length of the largest connected component of the California road network (with approximation) is: {np.mean(avg)} \n')  


def remove_random_elements(graph, remove_fraction_nodes=0, remove_fraction_edges=0):

    num_nodes_to_remove = int(len(graph.nodes) * remove_fraction_nodes)
    num_edges_to_remove = int(len(graph.edges) * remove_fraction_edges)

    nodes_to_remove = random.sample(list(graph.nodes), num_nodes_to_remove)
    graph.remove_nodes_from(nodes_to_remove)

    edges_to_remove = random.sample(list(graph.edges), num_edges_to_remove)
    graph.remove_edges_from(edges_to_remove)

    return graph


def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.kamada_kawai_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

def community_layout(g, partition):

    pos_communities = _position_communities(g, partition, scale=3.)
    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def select_connected_nodes(graph, num_nodes):

    if len(graph) < num_nodes:
        return None

    start_node = random.choice(list(graph.nodes))
    bfs_nodes = list(nx.bfs_tree(graph, start_node).nodes)
    selected_nodes = bfs_nodes[:min(num_nodes, len(bfs_nodes))]

    return selected_nodes

def draw_louvain_dendrogram(graph):
    """
    Draws a dendrogram representing the Louvain community structure of a graph.

    Args:
        graph: A NetworkX graph object.
    """
    dendrogram_data = community_louvain.generate_dendrogram(graph)
    
    # Extract the level assignments from the dendrogram
    level_assignments = [community_louvain.partition_at_level(dendrogram_data, level) for level in range(len(dendrogram_data))]
    
    # Create a distance matrix based on the level assignments
    num_nodes = len(graph.nodes)
    distance_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            for level in level_assignments:
                if level[list(graph.nodes)[i]] != level[list(graph.nodes)[j]]:
                    distance_matrix[i][j] = len(level_assignments) - level_assignments.index(level)
                    distance_matrix[j][i] = distance_matrix[i][j]
                    break 
                    
    # Convert to condensed distance matrix
    condensed_distance_matrix = squareform(distance_matrix)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance_matrix, method='ward')
    
    # Plot the dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=list(graph.nodes))
    plt.xlabel("Nodes")
    plt.ylabel("Distance")
    plt.title("Dendrogram of Louvain Community Structure")
    plt.savefig("louvain_dendrogram.png")
    plt.show()

def plot_lower_convex_hull(x_coords, y_coords):
    """
    Calculates and plots the lower convex hull of a set of 2D points.

    Args:
    x_coords (list): A list of x-coordinates (floats).
    y_coords (list): A list of y-coordinates (floats).
    """
    points = np.array(list(zip(x_coords, y_coords)))

    # Calculate the full convex hull
    hull = ConvexHull(points)

    # Find indices of lower convex hull vertices (based on x-coordinate)
    min_x_idx = np.argmin(points[hull.vertices, 0])
    max_x_idx = np.argmax(points[hull.vertices, 0])

    # Extract the lower hull vertices from the full convex hull
    if min_x_idx <= max_x_idx:
        lower_hull_vertices_indices = hull.vertices[min_x_idx : max_x_idx+1]
    else:
        lower_hull_vertices_indices = np.concatenate((hull.vertices[min_x_idx:], hull.vertices[:max_x_idx+1]))
    
    lower_hull_points = points[lower_hull_vertices_indices]

    plt.plot(points[:, 0], points[:, 1], 'o', linestyle = 'None')
    plt.plot(lower_hull_points[:, 0], lower_hull_points[:, 1], '-')
    plt.plot(lower_hull_points[:, 0], lower_hull_points[:, 1], 'o')

    plt.xlabel("log(Community Size)")
    plt.ylabel("log(Conductance)")
    plt.title("Network Community Profile Plot")
    plt.savefig("community_profile_roadNet-CA" + ".png")
    plt.show()



if __name__ == "__main__":

    '''''
    G = ox.graph_from_place('Los Angeles, California, USA', network_type='drive', simplify=True)
    ox.plot_graph(G)


    G = ox.graph_from_place('San Jose, California, USA', network_type='drive', simplify=True)
    ox.plot_graph(G)
    '''''

    file_path = "roadNet-CA.txt"
    graph = create_graph_from_edge_list(file_path)

    # print the number of nodes and edges and confirm that the execution is running properly
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())

    print("The original graph is directed:", nx.is_directed(graph))
    print("The original graph is weakly connected:", nx.is_weakly_connected(graph))
    print("The original graph is strongly connected:", nx.is_strongly_connected(graph))
    print("The reciprocity of the original graph is: ", nx.reciprocity(graph))

    graph = create_undirected_graph_from_edge_list(file_path)
    print("\n", end="")
    
    print("The number of nodes in the California road network is: ", graph.number_of_nodes())
    print("The number of edges in the California road network is: ", graph.number_of_edges())
    print("The California road network is directed:", nx.is_directed(graph))
    print("The California road network is connected:", nx.is_connected(graph))
    print("The degree assortativity coefficient of the California road network is: ", nx.degree_assortativity_coefficient(graph))

    result = dict(graph.degree()).values()
    data = list(result)
    numpyArray = np.array(data) 
    print("The average degree of the California road network is: ", np.mean(numpyArray))
    print("The average degree connectivity of the California road network is: ", nx.average_degree_connectivity(graph))
    print("The density of the California road network is: ", nx.density(graph))
    print("The average clustering coefficient of the California road network is: ", nx.average_clustering(graph))
    print("The number of connected components in the California road network is: ", nx.number_connected_components(graph))
    print("The transitivity of the California road network is: ", nx.transitivity(graph))
    print("The number of triangles in the California road network is: ", sum(nx.triangles(graph).values()) / 3)


    # plot the degree histogram/degree distribution of the graph
    fig = plt.figure("Degree of the California road network", figsize=(10, 10))
    axgrid = fig.add_gridspec(5, 4)
    degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)
    degree_distribution = np.array(degree_sequence) / np.sum(degree_sequence)
    ax0 = fig.add_subplot(axgrid[0:3, :])
    ax0.bar(*np.unique(degree_sequence, return_counts=True))
    ax0.set_xlabel("Degree")
    ax0.set_ylabel("Number of Nodes")
    plt.xticks(np.arange(min(degree_sequence) - 1, max(degree_sequence) + 1, 1.0))
    plt.xticks(rotation=-45)
    plt.title("Degree Histogram of the California road network")
    plt.savefig("degree_histogram_roadNet-CA" + ".png")
    plt.show()

    cc_generator = nx.connected_components(graph)
    lcc = max(cc_generator, key=len)
    lcc = graph.subgraph(lcc)

    # plot the degree distribution of a random graph

    n = lcc.number_of_nodes()
    m = lcc.number_of_edges() 
    seed = 1234
    rGraph = nx.gnm_random_graph(n, m, seed=seed, directed=False)
    fig = plt.figure("Random Graph", figsize=(10, 10))
    axgrid = fig.add_gridspec(5, 4)
    degree_sequence_random = sorted((d for n, d in rGraph.degree()), reverse=True)
    ax0 = fig.add_subplot(axgrid[0:3, :])
    ax0.bar(*np.unique(degree_sequence_random, return_counts=True))
    ax0.set_xlabel("Degree")
    ax0.set_ylabel("Number of Nodes")
    plt.xticks(np.arange(min(degree_sequence_random) - 1, max(degree_sequence_random) + 1, 1.0))
    plt.xticks(rotation=-45)
    plt.title("Degree Histogram of a random graph")
    plt.savefig("degree_histogram_random" + ".png")
    plt.show()

    # plot the Log log degree distribution of the graph
    degrees = dict(graph.degree())
    pos_degree_vals = list(filter(lambda val: val > 0, degrees.values()))
    uq_pos_degree_vals = sorted(set(pos_degree_vals))
    hist = [pos_degree_vals.count(x) for x in uq_pos_degree_vals]
    x = np.asarray(uq_pos_degree_vals, dtype = float)
    y = np.asarray(hist, dtype = float)
    logx = np.log(x)
    logy = np.log(y)

    my_pwlf = pwlf.PiecewiseLinFit(logx, logy)
    res = my_pwlf.fitfast(2)
    logy_hat = my_pwlf.predict(logx)

    plt.figure(figsize=(8,6))
    plt.xlim(min(logx), max(logx))
    plt.xlabel('log (Degree)')
    plt.ylabel('log (Number of nodes)')
    plt.title('Degree Distribution of network')
    a, b = np.polyfit(logx, logy, 1)
    scatter_plot = plt.plot(logx, logy, 'o')
    scatter_plot_regression = plt.plot(logx, a*logx + b)
    plt.plot(logx, logy_hat, '-', label='Piecewise Linear Fit')
    plt.savefig("Loglogdegree_graph_roadNet-CA" + ".png")
    plt.show()

    graph_entropy = entropy(degree_distribution)
    print(f"Graph entropy: {graph_entropy}")

    data = np.array(degree_sequence)
    fit = powerlaw.Fit(data, discrete=True, xmin=0)
    alpha = fit.power_law.alpha
    xmin = 0
    print(f"Power law exponent: {alpha}")
    print(f"Minimum value of x: {xmin}")
    distributions_to_compare = ['exponential', 'lognormal', 'lognormal_positive', 'stretched_exponential'] 
    for distribution in distributions_to_compare:
        R, p = fit.distribution_compare('power_law', distribution)
        print(f"Comparison against {distribution}: R = {R}, p = {p}") 

    data = np.array(degree_sequence)
    fit = powerlaw.Fit(data, discrete=True, xmin=3)
    alpha = fit.power_law.alpha
    xmin = 3
    print(f"Power law exponent: {alpha}")
    print(f"Minimum value of x: {xmin}")
    distributions_to_compare = ['exponential', 'lognormal', 'lognormal_positive', 'stretched_exponential'] 
    for distribution in distributions_to_compare:
        R, p = fit.distribution_compare('power_law', distribution)
        print(f"Comparison against {distribution}: R = {R}, p = {p}") 


    print("The number of nodes in the largest connected component of the California road network is: ", lcc.number_of_nodes())
    print("The number of edges in the largest connected component of the California road network is: ", lcc.number_of_edges())

    # calculate the clustering coefficient of the largest connected component of the California road network against a random graph
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    clustering_coeff = []
    for i in range(NUMITERS):
        randomGraph = nx.gnm_random_graph(n, m, seed=123*(i+1), directed=False)
        cc_generator = nx.connected_components(randomGraph)
        lcc_randomGraph = max(cc_generator, key=len)
        lcc_randomGraph = randomGraph.subgraph(lcc_randomGraph)
        clustering_coeff.append(nx.average_clustering(lcc)/max(nx.average_clustering(lcc_randomGraph), 1e-10))
        print("Done with batch", i+1, "of 30 for clustering coefficient")
    clusteringCoeff_mu = np.mean(clustering_coeff)
    clusteringCoeff_std = np.std(clustering_coeff)
    clustering_coeff_num = NUMITERS
    z = (clusteringCoeff_mu - 1) / (clusteringCoeff_std/np.sqrt(clustering_coeff_num))
    pvalue = 1 - stats.norm.cdf(z)
    print("The p-value of the clustering coefficient of the lcc from California road network against the lcc from a random graph by the one sample z-test is: ", pvalue)


    # plot the clustering coefficient of the graph
    n = graph.number_of_nodes()
    k = 3
    p_ws = 0.1
    wsGraph = nx.watts_strogatz_graph(n, k, p_ws, seed=1234)
    cc_generator_ws = nx.connected_components(wsGraph)
    lcc_ws = max(cc_generator_ws, key=len)
    lcc_ws = wsGraph.subgraph(lcc_ws)
    clustering_coeff_ws = nx.average_clustering(lcc_ws)
    print("The clustering coefficient of the largest connected component of a Watts-Strogatz graph is: ", clustering_coeff_ws)

    
    write_nodes_number_and_shortest_paths(lcc)

    # plot the histogram of the sizes of the communities of the random graph
    n = int(lcc.number_of_nodes()/50)
    p = lcc.number_of_edges()*1.0/(lcc.number_of_nodes()*(lcc.number_of_nodes()-1)/2)
    seed = 1234
    erGraph = nx.erdos_renyi_graph(n, p, seed=seed, directed=False)
    cc_generator_ER = nx.connected_components(erGraph)
    lccER = max(cc_generator_ER, key=len) 
    lccER = erGraph.subgraph(lccER)
    partitionER = nx.community.louvain_communities(lccER)
    print("The number of communities in the largest connected component of the random graph is: ", len(partitionER))

    community_sizesER = [len(c) for c in partitionER]
    plt.figure(figsize=(12,8))
    plt.hist(community_sizesER, bins=50, density=True)
    plt.xlabel("Community size")
    plt.ylabel("Number of communities")
    plt.title("Histogram of the sizes of the communities in the largest connected component of the random graph by Louvain algorithm")
    plt.plot(x, 'c', linewidth=2)
    plt.savefig("community_sizes_louvain_randomGraph" + ".png")
    plt.show()

    # apply the Louvain algorithm to detect communities in the graph


    partitionNetX = nx.community.louvain_communities(lcc)
    print("The number of communities in the largest connected component of the California road network is: ", len(partitionNetX))    

    # plot the histogram of the sizes of the communities
    community_sizes = [len(c) for c in partitionNetX]
    plt.figure(figsize=(12,8))
    plt.hist(community_sizes, bins=50, density=True)
    plt.xlabel("Community size")
    plt.ylabel("Number of communities")
    plt.title("Histogram of the sizes of the communities in the largest connected component of the California road network by Louvain algorithm")
    mu, std = stats.norm.fit(community_sizes)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'c', linewidth=2)
    plt.savefig("community_sizes_louvain_roadNet-CA" + ".png")
    plt.show()

    modularity = nx.community.modularity(lcc, partitionNetX)
    print("The modularity of the California road network is: ", modularity) 

    lccNK = nk.nxadapter.nx2nk(lcc)

    diam = nk.distance.Diameter(lccNK,algo=1)
    diam.run()
    diameter = diam.getDiameter()
    print("The diameter of the largest connected component of the California road network is: ", diameter)

    mc = nk.clique.MaximalCliques(lccNK)
    mc.run()
    maxCliques = mc.getCliques()
    clique_sizes = [len(clique) for clique in maxCliques]

    print("The maximum number of maximal cliques in the largest connected component of the California road network is: ", len(maxCliques))
    print("The maximum size of the maximal cliques in the largest connected component of the California road network is: ", max(clique_sizes))

    BC = nk.centrality.EstimateBetweenness(lccNK, 500)
    BC.run()
    betweenness = BC.scores()
    print("The average betweenness centrality of the largest connected component of the California road network is: ", sum(betweenness)/len(betweenness))
    print("The maximum betweenness centrality of the largest connected component of the California road network is: ", max(betweenness))

    CC = nk.centrality.ApproxCloseness(lccNK, 500)
    CC.run()
    closeness = CC.scores()
    print("The average closeness centrality of the largest connected component of the California road network is: ", statistics.mean(closeness))
    print("The maximum closeness centrality of the largest connected component of the California road network is: ", max(closeness))

    # plot the community layout of the graph

    community_to_color ={
        0 : 'xkcd:blue',
        1 : 'xkcd:orange',
        2 : 'xkcd:green',
        3 : 'xkcd:red',
        4 : 'xkcd:purple',
        5 : 'xkcd:brown',
        6 : 'xkcd:pink',
        7 : 'xkcd:gray',
        8 : 'xkcd:olive',
        9 : 'xkcd:cyan',
        10 : 'xkcd:light blue',
        11 : 'xkcd:light green',
        12 : 'xkcd:light red',
        13 : 'xkcd:light aqua',
        14 : 'xkcd:light lime green',
        15 : 'xkcd:goldenrod',
        16 : 'xkcd:faded blue',
        17 : 'xkcd:apricot',
        18 : 'xkcd:pale lavender',
        19 : 'xkcd:sunshine yellow',
        20 : 'xkcd:light purple',
        21 : 'xkcd:light pink',
        22 : 'xkcd:celadon',
        23 : 'xkcd:carolina blue',
        24 : 'xkcd:night blue', 
    }
    
    connected_nodes = select_connected_nodes(lcc, 1000)
    subgraph = lcc.subgraph(connected_nodes)
    node_to_community = community_louvain.best_partition(subgraph)
    pos = community_layout(subgraph, node_to_community)
    nx.draw(subgraph, pos, node_color=[community_to_color[node_to_community[node]%25] for node in subgraph.nodes()], with_labels=False, node_size=30, font_size=8)
    plt.title("Community Layout of the California Road Network")
    plt.axis('off') 
    plt.savefig("community_layout_roadNet-CA" + ".png")
    plt.show()  
    
    connected_nodes_den = select_connected_nodes(lcc, 100) 
    subgraph_den = lcc.subgraph(connected_nodes_den)
    draw_louvain_dendrogram(subgraph_den)

    partition = community_louvain.best_partition(lcc)

    # Calculate conductance for each community
    conductance_values = []
    community_sizes = []

    print("Calculating conductance for each community...")
    
    for com in set(partition.values()):
        community_nodes = [nodes for nodes in partition if partition[nodes] == com]
        if len(community_nodes) > 1:
            subgraph = lcc.subgraph(community_nodes)
            cut_size = nx.cut_size(lcc, community_nodes)
            degree_sum = sum(dict(lcc.degree(community_nodes)).values())
            conductance = cut_size / degree_sum if degree_sum > 0 else 0
        else:
            conductance = 0
        conductance_values.append(math.log(conductance))
        community_sizes.append(math.log(len(community_nodes)))

    # Plot community profile
    x = np.array(community_sizes)
    y = np.array(conductance_values)

    plot_lower_convex_hull(x, y)



    # Simulate percolation attack


    largest_components_UP_array = np.zeros((10, 11))
    largest_components_TA_array = np.zeros((10, 11))
    betweenness_UP_array = np.zeros((10, 11))
    betweenness_TA_array = np.zeros((10, 11))

    for i in range(10):
        connected_nodes = select_connected_nodes(lcc, 10000)
        lcc_UP = lcc.subgraph(connected_nodes)
        lcc_TA = lcc.subgraph(connected_nodes)

        fraction_target = 0.3
        attack_fractions = np.linspace(0, fraction_target, 11)  # From 0% to 30% removal
        uniformPercolation = lcc_UP.copy()
        freeze = nx.freeze(uniformPercolation)
        uniformPercolation = nx.Graph(freeze)
        num_nodes_to_remove_per_cycle = int(len(uniformPercolation.nodes()) * fraction_target/10)

        targetedAttack = lcc_TA.copy()
        freeze = nx.freeze(targetedAttack)
        targetedAttack = nx.Graph(freeze)   
        targetedAttackNK = nk.nxadapter.nx2nk(targetedAttack)
        BC = nk.centrality.Betweenness(targetedAttackNK)
        BC.run()
        betweennessNK = BC.ranking()
        num_nodes_to_remove_per_cycle = int(targetedAttackNK.numberOfNodes() * fraction_target/10)
        num_nodes_to_remove_total = int(targetedAttackNK.numberOfNodes() * fraction_target)
        list_of_nodes_targeted_attack = [betweennessNK[i][0] for i in range(num_nodes_to_remove_total)]


        largest_components_UP_array[i][0] = len(uniformPercolation.nodes())
        cc_generator = nx.connected_components(uniformPercolation)
        percolationLCC = max(cc_generator, key=len)
        percolation_LCC = uniformPercolation.subgraph(percolationLCC)
        PercolationLCC_NK = nk.nxadapter.nx2nk(percolation_LCC)
        BC = nk.centrality.EstimateBetweenness(PercolationLCC_NK, 200)
        BC.run()
        betweennessNK = BC.scores()
        betweenness_UP_array[i][0] = sum(betweennessNK)/len(betweennessNK)

        largest_components_TA_array[i][0] = len(targetedAttack.nodes())
        cc_generator = nx.connected_components(targetedAttack)
        targetedAttackLCC = max(cc_generator, key=len)
        targetedAttack_LCC = targetedAttack.subgraph(targetedAttackLCC)
        targetedAttackLCC_NK = nk.nxadapter.nx2nk(targetedAttack_LCC)
        BC = nk.centrality.EstimateBetweenness(targetedAttackLCC_NK, 200)
        BC.run()
        betweennessNK = BC.scores()
        betweenness_TA_array[i][0] = sum(betweennessNK)/len(betweennessNK)


        j = 1
        for _ in range(10):
            nodes_to_remove = random.sample(list(uniformPercolation.nodes()), num_nodes_to_remove_per_cycle)
            for node in nodes_to_remove:
                if node in uniformPercolation.nodes():
                    uniformPercolation.remove_node(node) 

            cc_generator = nx.connected_components(uniformPercolation)
            percolationLCC = max(cc_generator, key=len)
            percolation_LCC = uniformPercolation.subgraph(percolationLCC)
            size = percolation_LCC.number_of_nodes()
            largest_components_UP_array[i][j] = size

            PercolationLCC_NK = nk.nxadapter.nx2nk(percolation_LCC)
            BC = nk.centrality.EstimateBetweenness(PercolationLCC_NK, 200)
            BC.run()
            betweennessNK = BC.scores()
            betweenness_UP_array[i][j] = sum(betweennessNK)/len(betweennessNK)


            nodes_to_remove = list_of_nodes_targeted_attack[j*num_nodes_to_remove_per_cycle:(j+1)*num_nodes_to_remove_per_cycle]
            
            for node in nodes_to_remove:
                if targetedAttackNK.hasNode(node):
                    targetedAttackNK.removeNode(node)

            cc = nk.components.ConnectedComponents(targetedAttackNK)
            cc.run()
            targetedAttackLCC_NK = cc.extractLargestConnectedComponent(targetedAttackNK, compactGraph=True)
            size = targetedAttackLCC_NK.numberOfNodes()
            largest_components_TA_array[i][j] = size

            BC = nk.centrality.EstimateBetweenness(targetedAttackLCC_NK, 100)
            BC.run()
            betweenness = BC.scores()
            betweenness_TA_array[i][j] = sum(betweenness)/len(betweenness)


            j += 1        
        print("Done with batch", i+1, "of 10 batches for percolation attack simulation")


    lcc_UP_mu = list(np.mean(largest_components_UP_array, axis=0))
    lcc_UP_std = list(np.std(largest_components_UP_array, axis=0))
    betweenness_UP_mu = list(np.mean(betweenness_UP_array, axis=0))
    betweenness_UP_std = list(np.std(betweenness_UP_array, axis=0))   


    lcc_TA_mu = list(np.mean(largest_components_TA_array, axis=0))
    lcc_TA_std = list(np.std(largest_components_TA_array, axis=0))
    betweenness_TA_mu = list(np.mean(betweenness_TA_array, axis=0))
    betweenness_TA_std = list(np.std(betweenness_TA_array, axis=0))


    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.errorbar(attack_fractions, lcc_UP_mu, yerr=lcc_UP_std, label='Uniform Percolation')
    plt.errorbar(attack_fractions, lcc_TA_mu, yerr=lcc_TA_std, label='Targeted Attack')
    plt.title('Size of the Largest Connected Component vs Attack Fraction')
    plt.xlabel('Attack Fraction')
    plt.ylabel('Size of Largest Connected Component')
    plt.legend(['Uniform Percolation', 'Targeted Attack'])
    plt.savefig("largest_connected_component_roadNet-CA" + ".png")
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.errorbar(attack_fractions, betweenness_UP_mu, yerr=betweenness_UP_std, label='Uniform Percolation')
    plt.errorbar(attack_fractions, betweenness_TA_mu, yerr=betweenness_TA_std, label='Targeted Attack')
    plt.title('Average Betweenness Centrality vs Attack Fraction')
    plt.xlabel('Attack Fraction')
    plt.ylabel('Average Betweenness Centrality')
    plt.legend(['Uniform Percolation', 'Targeted Attack'])
    plt.savefig("betweenness_roadNet-CA" + ".png")
    plt.show()


    print("\n", end="")