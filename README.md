# Overview

Based on a given dataset, `main.py` script can:
- Model two types of graphs: 
  - The `SNet` network contains subreddits as nodes. Two subreddits are connected if there are at least two users active on both subreddits. Weight corresponds to the number of active users between the subreddits, and
  - The `UserNet` network contains users as nodes. There is an edge between user A and user B (directed) if user A commented on user B's submission or comment. Weight corresponds to the number of times user A has commented user B's submission or comment.
- Compute general information about the graphs: density, diameter, average distance between two nodes, etc.
- Plot various distributions: node degree distribution, edge weight distribution, etc.
- Do centrality analysis: degree centrality, closeness centrality, betweenness centrality, eigenvector centrality and *Katz*'s centrality.
- Run community detection algorithms: *Louvain*, *Girvan-Newman* and dendrogram analysis.

# Implementation Details

- *networkx* package was used to model and compute various information about the graphs.
- *Gephi* tool was used for visualization of graphs.
