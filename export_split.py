import pandas as pd
import torch

# load graphs and saved split
graphs = torch.load(
    "/Users/harshithabhupathiraju/ABIDE_Project/graphs/abide_cc200_graphs.pt",
    weights_only=False
)
split = torch.load(
    "/Users/harshithabhupathiraju/ABIDE_Project/graphs/split_indices.pt",
    weights_only=False
)

# create subject ID and split table
rows = []

# training subjects
for idx in split["train"]:
    graph = graphs[idx]
    rows.append({
        "SUB_ID": int(graph.subject_id),
        "split": "train"
    })


# validation subjects
for idx in split["val"]:
    graph = graphs[idx]
    rows.append({
        "SUB_ID": int(graph.subject_id),
        "split": "validation"
    })

# test subjects
for idx in split["test"]:
    graph = graphs[idx]
    rows.append({
        "SUB_ID": int(graph.subject_id),
        "split": "test"
    })


# convert to dataframe
split_df = pd.DataFrame(rows)


# split sizes
print("Split sizes:")
print(split_df["split"].value_counts())

# first 10 rows
print("\nFirst 10 rows:")
print(split_df.head(10))


# save for R
output_path = (
    "/Users/harshithabhupathiraju/"
    "ABIDE_Project/graphs/model_splits.csv"
)
split_df.to_csv(
    output_path,
    index=False
)
print("\nSaved to:")
print(output_path)

print(split_df["split"].value_counts())


# check asd and control distribution using phenotype data
pheno = pd.read_csv(
    "/Users/harshithabhupathiraju/ABIDE_Project/abide_data/"
    "ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
)
pheno = pheno[["SUB_ID", "DX_GROUP"]]

pheno["DX_GROUP"] = pheno["DX_GROUP"].apply(  # asd = 1, control = 0
    lambda x: 1 if x == 1 else 0
)

# merge phenotype with split information
check = split_df.merge(
    pheno,
    on="SUB_ID",
    how="left"
)

print("\nClass distribution:")
print(
    pd.crosstab(
        check["split"],
        check["DX_GROUP"]
    )
)