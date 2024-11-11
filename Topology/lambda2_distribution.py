from tqdm.contrib.concurrent import process_map
from my_graphs_dataset import GraphDataset, GraphType
import networkx as nx
import numpy as np
import plotly.express as px
import pandas as pd
from scipy.stats import zscore


def algebraic_connectivity(G):
    L = nx.laplacian_matrix(G).toarray()
    lambdas = sorted(np.linalg.eigvalsh(L))
    return lambdas[1]


def norm_algebraic_connectivity(G, N):
    return algebraic_connectivity(G) / N


def spectral_radius(G):
    L = nx.laplacian_matrix(G).toarray()
    lambdas = np.linalg.eigvalsh(L)
    return max(abs(lambdas))


def worker(graph):
    G = GraphDataset.parse_graph6(graph)
    num_nodes = G.number_of_nodes()
    l2 = algebraic_connectivity(G)
    return l2, num_nodes


selection = {
    3: -1,
    4: -1,
    5: -1,
    6: -1,
    7: -1,
    8: -1,
    9: 10000,
    10: 10000,
    GraphType.RANDOM_MIX: (10000, range(11, 21)),
}
loader = GraphDataset(selection=selection)
all_results = []


# If batch_size="auto", loader yields all graphs from individual files.
for graphs in loader.graphs(raw=True, batch_size=10000):
    # Process map runs the multiprocessing pool and displays a progress bar with tqdm.
    result = process_map(worker, graphs, chunksize=1000)
    all_results.extend(result)

# Prepare data for plotting
df = pd.DataFrame(all_results, columns=['algebraic_connectivity', 'num_nodes'])

# Perform z-score normalization within each group of num_nodes
# df['algebraic_connectivity'] = df.groupby('num_nodes')['algebraic_connectivity'].transform(zscore)

# Create ridgeline plot
fig = px.violin(df, x='algebraic_connectivity', y='num_nodes', points=False, box=True)
fig.update_traces(orientation='h', side='positive', width=3, spanmode="hard")
fig.show()