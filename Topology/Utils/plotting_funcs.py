import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
import numpy as np


def plotly_nx(G):
    pos = nx.kamada_kawai_layout(G)
    nx.set_node_attributes(G, pos, 'pos')

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            showlegend=False,
                            line=dict(width=2, color='#888'),
                            hoverinfo='none',
                            mode='lines')

    node_x = []
    node_y = []
    node_t = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_t.append(node)

    node_trace = go.Scatter(x=node_x, y=node_y,
                            mode='markers+text',
                            text=node_t,
                            textposition="bottom center",
                            hoverinfo='none',
                            marker=dict(
                                color='blue',
                                size=15,
                                line_width=0),
                            showlegend=False)

    return edge_trace, node_trace
