"""
Generate all connected graphs.

This module defines functions for generating all possible graphs with a given
number of nodes; then selects from them all graphs that are 1) connected,
2) not isomorphic to other graphs in the set. The complexity is O(e^(N^2)).
"""
import itertools
from collections import defaultdict
import numpy as np
import networkx as nx
import codetiming
from pathlib import Path


def make_graphs(n=2, i=None, j=None):
    """Make a graph recursively, by either including, or skipping each edge.
    Edges are given in lexicographical order by construction."""
    out = []
    if i is None:  # First call
        out = [[(0, 1)] + r for r in make_graphs(n=n, i=0, j=1)]
    elif j < n - 1:
        out += [[(i, j + 1)] + r for r in make_graphs(n=n, i=i, j=j + 1)]
        out += [r for r in make_graphs(n=n, i=i, j=j + 1)]
    elif i < n - 1:
        out = make_graphs(n=n, i=i + 1, j=i + 1)
    else:
        out = [[]]
    return out


def permute(g, n):
    """Create a set of all possible isomorphic codes for a graph,
    as nice hashable tuples. All edges are i<j, and sorted lexicographically."""
    # ps = perm(n)
    out = set([])
    for p in itertools.permutations(range(n), n):
        out.add(tuple(sorted([(p[i], p[j]) if p[i] < p[j] else (p[j], p[i]) for i, j in g])))
    return list(out)


def connected(g):
    """Check if the graph is fully connected, with Union-Find."""
    nodes = set([i for e in g for i in e])
    roots = {node: node for node in nodes}

    def _root(node, depth=0):
        if node == roots[node]:
            return (node, depth)
        else:
            return _root(roots[node], depth + 1)

    for i, j in g:
        ri, di = _root(i)
        rj, dj = _root(j)
        if ri == rj: continue
        if di <= dj:
            roots[ri] = rj
        else:
            roots[rj] = ri
    return len(set([_root(node)[0] for node in nodes])) == 1


def filter(gs, target_nv, do_checks=True):
    """
    Filter all improper graphs: those with not enough nodes,
    those not fully connected, and those isomorphic to previously considered.
    """
    mem = set({})
    gs2 = []
    for g in gs:
        if do_checks:
            nv = len(set([i for e in g for i in e]))
            if nv != target_nv:
                continue
            if not connected(g):
                continue
        if tuple(g) not in mem:
            gs2.append(g)
            mem |= set(permute(g, target_nv))
        else:
            continue
        # print('\n'.join([str(a) for a in mem]))
    return gs2

def filter2(gs, target_nv, do_checks=True):
    """
    Filter all improper graphs: those with not enough nodes,
    those not fully connected, and those isomorphic to previously considered.
    """
    mem = defaultdict(list)
    gs2 = []
    for g in gs:
        if do_checks:
            nv = len(set([i for e in g for i in e]))
            if nv != target_nv:
                continue
            if not connected(g):
                continue
        
        nx_g = nx.from_edgelist(g)
        for m in mem[len(g)]:
            if nx.is_isomorphic(nx_g, m):
                break
        else:
            mem[len(g)].append(nx_g)
            gs2.append(g)
            
    return gs2



def make_graphs_filtered(NV):
    print(f'Building graphs with {NV} nodes...', end=' ')
    gs_raw = make_graphs(NV)
    print('Filtering...', end=' ')
    gs = filter(gs_raw, NV)
    print('Done')

    return gs


def make_graphs_faster(NV, old_gs=None):
    if old_gs is None:
        return make_graphs_filtered(NV)
    
    print(f'Building graphs with {NV} nodes...', end=' ')
    gs_raw = []
    for g in old_gs:
        for i in range(NV-1):
            for comb in itertools.combinations(range(NV-1), i+1):    
                temp = []
                for c in comb:
                    temp.append((c, NV-1))
                gs_raw.append(sorted(g + temp))

    print('Filtering...', end=' ')
    gs = filter2(gs_raw, NV, do_checks=False)
    print('Done')
    
    return sorted(gs, key=lambda x: len(x))

def make_graphs_very_fast(NV):
    # call and parse: geng -c 3 | listg -e
    pass


def main():
    data_folder = Path.cwd() / 'Data'

    total = 0
    gs = None

    for i in range(3, 8 + 1):
        all_graphs = {}
        with codetiming.Timer():
            gs = make_graphs_faster(i, gs)
        print(f"Graphs built: {len(gs)}\n")
        total += len(gs)

        for k, edges in enumerate(gs):
            g = nx.from_edgelist(edges)
            all_graphs[f'G{i},{k}'] = nx.to_numpy_array(g)

        with open(data_folder / f'UniqueGraphs_{i}.npz', 'wb') as stream:
            np.savez(stream, **all_graphs)

    print(f"Files generated. Total {total} graphs.")

if __name__ == '__main__':
    main()