import numpy as np
import torch
from nilearn import datasets
from torch_geometric.data import Data

abide = datasets.fetch_abide_pcp(
    data_dir="/Users/harshithabhupathiraju/ABIDE_Project/abide_data",
    pipeline="cpac",
    derivatives=["rois_cc200"],
    quality_checked=True
)

ts = abide.rois_cc200[0] # one participant
print("Time series shape:", ts.shape)


# functional connectivity
corr_matrix = np.corrcoef(ts, rowvar=False)
print("Correlation matrix shape:", corr_matrix.shape)


# get rid of self-connections
np.fill_diagonal(corr_matrix, 0)


# create edges
threshold = 0.2

rows, cols = np.where(np.abs(corr_matrix) >= threshold)

edge_index = torch.tensor(
    np.vstack([rows, cols]),
    dtype=torch.long
)

edge_weight = torch.tensor(
    corr_matrix[rows, cols],
    dtype=torch.float
)

# node features
x = torch.tensor(
    corr_matrix,
    dtype=torch.float
)

# pytorch graph
graph = Data(
    x=x,
    edge_index=edge_index,
    edge_weight=edge_weight
)

print("\nGraph:")
print(graph)
print("\nNode features:")
print(graph.x.shape)
print("\nEdges:")
print(graph.edge_index.shape)
print("\nEdge weights:")
print(graph.edge_weight.shape)