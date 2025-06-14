import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
from networkx.algorithms.community import greedy_modularity_communities

def load_data(csv_path):
    """
    Load CSV and preprocess keywords into lists.
    """
    df = pd.read_csv(csv_path)
    df['keyword_list'] = df['keywords'].apply(
        lambda x: [kw.strip() for kw in x.split(',')] if isinstance(x, str) else []
    )
    return df

def build_graph(df, weight_threshold=5):
    """
    Build a co-occurrence graph from keyword lists.
    Only include edges with co-occurrence count >= weight_threshold.
    """
    edge_list = []
    for kws in df['keyword_list']:
        unique_kws = set(kws)
        for w1, w2 in combinations(unique_kws, 2):
            edge_list.append((w1, w2))
    edge_counts = Counter(edge_list)

    G = nx.Graph()
    for (u, v), w in edge_counts.items():
        if w >= weight_threshold:
            G.add_edge(u, v, weight=w)
    return G

def detect_communities(G):
    """
    Detect communities using greedy modularity.
    Returns the list of communities and a node->community index map.
    """
    communities = list(greedy_modularity_communities(G))
    node_community = {node: idx for idx, com in enumerate(communities) for node in com}
    return communities, node_community

def filter_top_nodes(G, top_n=30):
    """
    Filter graph to top_n nodes by degree centrality.
    """
    deg_cent = nx.degree_centrality(G)
    top_nodes = [n for n, _ in sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    return G.subgraph(top_nodes)

def visualize_network(G_sub, node_community, layout='kamada_kawai'):
    """
    Visualize the subgraph with node size/colour by community.
    Layout: 'kamada_kawai' or 'spring'
    """
    if layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G_sub)
    else:
        pos = nx.spring_layout(G_sub, k=0.15, seed=42)

    colors = [node_community.get(n, -1) for n in G_sub.nodes()]
    sizes = [G_sub.degree(n) * 200 for n in G_sub.nodes()]

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G_sub, pos, node_size=sizes, node_color=colors, cmap=plt.cm.tab20)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.5, width=1)
    nx.draw_networkx_labels(G_sub, pos, font_size=10)
    plt.title("Journal Keywords Network (Weight ≥ 5, KK Layout)")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Path to the journal CSV file
    csv_path = 'data/02_journal_2020_2025.csv'

    # Load and preprocess data
    df = load_data(csv_path)

    # Build the co-occurrence graph
    G = build_graph(df, weight_threshold=5)

    # Detect communities in the full graph
    communities, node_community = detect_communities(G)

    # Filter to top 30 nodes by centrality
    G_sub = filter_top_nodes(G, top_n=30)

    # Visualize using Kamada–Kawai layout
    visualize_network(G_sub, node_community, layout='kamada_kawai')
