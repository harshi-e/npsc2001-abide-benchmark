import torch
import numpy as np
from sklearn.model_selection import train_test_split

graph_path = "/Users/harshithabhupathiraju/ABIDE_Project/graphs/abide_cc200_graphs.pt"
graphs = torch.load(graph_path, weights_only=False)
print("Number of graphs:", len(graphs))

# separating NYU from other sites
nyu_indices = []
non_nyu_indices = []

for i, graph in enumerate(graphs):
    if graph.site_id == "NYU":
        nyu_indices.append(i)
    else:
        non_nyu_indices.append(i)

print("\nSite split:")
print("Non-NYU subjects:", len(non_nyu_indices))
print("NYU subjects:", len(nyu_indices))

# labels for non-NYU subjects
non_nyu_labels = [
    graphs[i].y.item()
    for i in non_nyu_indices
]

# split non-NYU into training and validation
train_indices, val_indices = train_test_split(
    non_nyu_indices,
    test_size=0.20,
    random_state=42,
    stratify=non_nyu_labels
)

# split sizes
print("\nFinal split:")
print("Training:", len(train_indices))
print("Validation:", len(val_indices))
print("NYU test:", len(nyu_indices))


# asd and control balance
def count_labels(indices):
    asd = sum(graphs[i].y.item() == 1 for i in indices)
    control = sum(graphs[i].y.item() == 0 for i in indices)
    return asd, control

train_asd, train_control = count_labels(train_indices)
val_asd, val_control = count_labels(val_indices)
test_asd, test_control = count_labels(nyu_indices)

print("")
print("Class distribution:")
print(f"Training:   ASD = {train_asd}, "f"Control = {train_control}")
print(f"Validation: ASD = {val_asd}, "f"Control = {val_control}")
print(f"NYU test:   ASD = {test_asd}, "f"Control = {test_control}")



split_path = "/Users/harshithabhupathiraju/ABIDE_Project/graphs/split_indices.pt"

torch.save({
        "train": train_indices,
        "val": val_indices,
        "test": nyu_indices
    },
    split_path
)

print("\nSaved split to:")
print(split_path)