import numpy as np
import torch
from nilearn import datasets
from torch_geometric.data import Data

# load ABIDE data
abide = datasets.fetch_abide_pcp(
    data_dir="/Users/harshithabhupathiraju/ABIDE_Project/abide_data",
    pipeline="cpac",
    derivatives=["rois_cc200"],
    quality_checked=True
)

print("Number of subjects:", len(abide.rois_cc200))


# phenotype information
phenotype = abide.phenotypic
print("\nPhenotype columns:")
print(phenotype.columns.tolist())


# build the graphs

graphs = []
threshold = 0.2

for i, ts in enumerate(abide.rois_cc200):
    print(f"Processing subject {i + 1}/{len(abide.rois_cc200)}")
    corr_matrix = np.corrcoef(ts, rowvar=False) # functional connectivity
    corr_matrix = np.nan_to_num(corr_matrix) # replace NaN
    np.fill_diagonal(corr_matrix, 0) # get rid of the self-connections

    # create edges
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

    # phenotype info
    subject_id = phenotype.iloc[i]["SUB_ID"]
    dx_group = phenotype.iloc[i]["DX_GROUP"]
    site_id = phenotype.iloc[i]["SITE_ID"]
    
    if dx_group == 1:
        label = 1
    else:
        label = 0

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.tensor([label], dtype=torch.long)
    )

    graph.subject_id = subject_id
    graph.site_id = site_id
    graphs.append(graph)



# check dataset

print("\nFinished!")

print("Number of graphs:", len(graphs))

print("\nFirst graph:")
print(graphs[0])

print("\nFirst graph label:")
print(graphs[0].y)

print("\nFirst graph site:")
print(graphs[0].site_id)

print("\nFirst graph subject:")
print(graphs[0].subject_id)


# count asd and control
labels = [graph.y.item() for graph in graphs]
print("\nNumber of ASD subjects:", sum(label == 1 for label in labels))
print("Number of Control subjects:", sum(label == 0 for label in labels))

print("\nFirst 10 subjects:")
for i in range(10):
    row = phenotype.iloc[i]
    print(
        i,
        "SUB_ID:", row["SUB_ID"],
        "DX_GROUP:", row["DX_GROUP"],
        "SITE_ID:", row["SITE_ID"],
        "Graph subject:", graphs[i].subject_id,
        "Graph label:", graphs[i].y.item(),
        "Graph site:", graphs[i].site_id
    )

import os
graph_dir = "/Users/harshithabhupathiraju/ABIDE_Project/graphs"
os.makedirs(graph_dir, exist_ok=True)
save_path = os.path.join(graph_dir, "abide_cc200_graphs.pt")
torch.save(graphs, save_path)

print("\nSaved graphs to:")
print(save_path)