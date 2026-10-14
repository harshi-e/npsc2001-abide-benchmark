import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool



graph_path = (
    "/Users/harshithabhupathiraju/ABIDE_Project/"
    "graphs/abide_cc200_graphs.pt"
)

graphs = torch.load(graph_path, weights_only=False)
print("Total graphs:", len(graphs))

split_path = (
    "/Users/harshithabhupathiraju/ABIDE_Project/"
    "graphs/split_indices.pt"
)

splits = torch.load(split_path, weights_only=False)

train_indices = splits["train"]
val_indices = splits["val"]
test_indices = splits["test"]


# create the three datasets using the existing fixed split
train_graphs = [graphs[i] for i in train_indices]
val_graphs = [graphs[i] for i in val_indices]
test_graphs = [graphs[i] for i in test_indices]


print("\nDataset sizes:")
print("Training:", len(train_graphs))
print("Validation:", len(val_graphs))
print("Test:", len(test_graphs))



# create validation and test dataloaders
val_loader = DataLoader(
    val_graphs,
    batch_size=16,
    shuffle=False
)
test_loader = DataLoader(
    test_graphs,
    batch_size=16,
    shuffle=False
)

# inspected one batch to check the graph's structure
inspection_loader = DataLoader(
    train_graphs,
    batch_size=16,
    shuffle=False
)

batch = next(iter(inspection_loader))

print("\nFirst training batch:")
print(batch)

print("\nBatch node features:")
print(batch.x.shape)

print("\nBatch edges:")
print(batch.edge_index.shape)

print("\nBatch edge weights:")
print(batch.edge_weight.shape)

print("\nBatch labels:")
print(batch.y.shape)
print(batch.y)

print("\nNumber of graphs in batch:")
print(batch.num_graphs)



# define the GCN

class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(
            in_channels=200,
            out_channels=64
        )
        self.conv2 = GCNConv(
            in_channels=64,
            out_channels=32
        )
        self.classifier = nn.Linear(
            in_features=32,
            out_features=2
        )

    def forward(self, x, edge_index, edge_weight, batch):
        edge_weight = torch.abs(edge_weight) # using edge magnitudes to avoid problems with negative edge weights

        # 1st graph convolution
        x = self.conv1(
            x,
            edge_index,
            edge_weight
        )
        x = F.relu(x)

        # 2ns graph convolution
        x = self.conv2(
            x,
            edge_index,
            edge_weight
        )
        x = F.relu(x)
        x = global_mean_pool(x, batch) # produce one representation per subject

        # ASD/control classification logits
        x = self.classifier(x)
        return x



# evaluation function

def evaluate_model(model, loader):
    model.eval()
    labels = []
    predictions = []
    probabilities = []
    inference_start = time.perf_counter() # time only the model's inference loop

    with torch.no_grad():
        for batch in loader:
            output = model(
                batch.x,
                batch.edge_index,
                batch.edge_weight,
                batch.batch
            )

            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            labels.extend(batch.y.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())
    inference_time = time.perf_counter() - inference_start

    # converting results into arrays
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    # calculate evaluation metrics
    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities
    )

    # classes: control = 0, asd = 1
    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    return {
        "accuracy": accuracy,
        "f1": macro_f1,
        "auc": roc_auc,
        "confusion_matrix": cm,
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "inference_time": inference_time
    }

# extract site id from each graph
def get_site_id(graph):
    for field in ["SITE_ID", "site_id"]:
        value = getattr(graph, field, None)
        if value is not None:
            if torch.is_tensor(value):
                value = value.item()
            return str(value)

    raise ValueError(
        "SITE_ID is missing from a graph. "
        "Check the graph attributes and attach site metadata "
        "before running the per-site analysis."
    )



# run 3 experiements
run_results = []

site_predictions = [] # store validation predictions for per-site analysis

seeds = [123, 456, 789]
num_epochs = 30

for run_seed in seeds:

    print("\n" + "#" * 60)
    print(f"Run with seed: {run_seed}")
    print("#" * 60)

    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    train_generator = torch.Generator()
    train_generator.manual_seed(run_seed)

    train_loader = DataLoader(
        train_graphs,
        batch_size=16,
        shuffle=True,
        generator=train_generator
    )

    model = GCN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    # reset best-model tracking for this run
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    best_model_state = None

    # start total training and validation timer
    training_start = time.perf_counter()


    # training and validation
    for epoch in range(num_epochs):

        # training

        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            output = model(
                batch.x,
                batch.edge_index,
                batch.edge_weight,
                batch.batch
            )
            loss = criterion(
                output,
                batch.y
            )
            loss.backward()
            optimizer.step()
            batch_size = batch.y.size(0)
            total_train_loss += loss.item() * batch_size
            predictions = output.argmax(dim=1)

            train_correct += (
                predictions == batch.y
            ).sum().item()
            train_total += batch_size

        train_loss = total_train_loss / train_total
        train_accuracy = train_correct / train_total


        # validation

        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                output = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_weight,
                    batch.batch
                )
                loss = criterion(
                    output,
                    batch.y
                )

                batch_size = batch.y.size(0)
                total_val_loss += loss.item() * batch_size
                predictions = output.argmax(dim=1)

                val_correct += (
                    predictions == batch.y
                ).sum().item()
                val_total += batch_size

        val_loss = total_val_loss / val_total
        val_accuracy = val_correct / val_total

        # save best checkpoint

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(
                model.state_dict()
            )

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

    training_time = time.perf_counter() - training_start # end timer after training and validation

    print("\nTraining and validation completed.")
    print(f"Total time: {training_time:.2f} seconds")
    print(
        f"Average time per epoch: "
        f"{training_time / num_epochs:.2f} seconds"
    )

    print("\nBest model:")
    print("Best epoch:", best_epoch)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(
        f"Validation accuracy at best epoch: "
        f"{best_val_accuracy:.4f}"
    )

    # IMPORTANT: restore the best checkpoint BEFORE evaluation
    model.load_state_dict(best_model_state)
    model.eval()


    # evaluate best model on validation set
    val_results = evaluate_model(
        model,
        val_loader
    )


    # save validation predictions for per-site analysis
    validation_site_ids = [
        get_site_id(graph)
        for graph in val_graphs
    ]

    site_predictions.append(
        pd.DataFrame({
            "Model": "GCN",
            "Seed": run_seed,
            "SITE_ID": validation_site_ids,
            "Actual": val_results["labels"],
            "Predicted": val_results["predictions"],
            "Probability": val_results["probabilities"]
        })
    )


    print("\n" + "=" * 50)
    print(f"Validation Results with Seed: {run_seed}")
    print("=" * 50)

    print(f"Accuracy: {val_results['accuracy']:.4f}")
    print(f"Macro F1: {val_results['f1']:.4f}")
    print(f"ROC-AUC: {val_results['auc']:.4f}")

    print("\nConfusion Matrix:")
    print(val_results["confusion_matrix"])

    print(
        "\nNumber of validation subjects:",
        len(val_results["labels"])
    )


    # evaluate best model on unseen NYU test set

    # NYU is used only for final evaluation, not model selection.
    test_results = evaluate_model(
        model,
        test_loader
    )

    print("\n" + "=" * 50)
    print(f"NYU Unseen Test Results with Seed: {run_seed}")
    print("=" * 50)
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Macro F1: {test_results['f1']:.4f}")
    print(f"ROC-AUC: {test_results['auc']:.4f}")
    print("\nConfusion Matrix:")
    print(test_results["confusion_matrix"])
    print(
        "\nNumber of test subjects:",
        len(test_results["labels"])
    )
    print(
        f"\nNYU inference time: "
        f"{test_results['inference_time']:.4f} seconds"
    )


    # store this run's results

    run_results.append({
        "seed": run_seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,

        "val_accuracy": val_results["accuracy"],
        "val_f1": val_results["f1"],
        "val_auc": val_results["auc"],
        "val_cm": val_results["confusion_matrix"],

        "test_accuracy": test_results["accuracy"],
        "test_f1": test_results["f1"],
        "test_auc": test_results["auc"],
        "test_cm": test_results["confusion_matrix"],

        "training_time": training_time,
        "inference_time": test_results["inference_time"]
    })


# summarise results across 3 runs

print("\n" + "=" * 60)
print("Results across 3 runs: mean +/- sd")
print("=" * 60)

metrics = [
    "val_accuracy",
    "val_f1",
    "val_auc",
    "test_accuracy",
    "test_f1",
    "test_auc",
    "training_time",
    "inference_time"
]

for metric in metrics:
    values = [
        result[metric]
        for result in run_results
    ]
    mean_value = np.mean(values)
    std_value = np.std(values, ddof=1)
    if metric in ["training_time", "inference_time"]:
        print(
            f"{metric}: "
            f"{mean_value:.4f} +/- {std_value:.4f} seconds"
        )
    else:
        print(
            f"{metric}: "
            f"{mean_value:.4f} +/- {std_value:.4f}"
        )


# show results from each run

print("\n" + "=" * 60)
print("Individual run results")
print("=" * 60)

for result in run_results:
    print(f"\nSeed: {result['seed']}")
    print("Best epoch:", result["best_epoch"])
    print(
        f"NYU Accuracy: {result['test_accuracy']:.4f} | "
        f"Macro F1: {result['test_f1']:.4f} | "
        f"ROC-AUC: {result['test_auc']:.4f}"
    )
    print("NYU confusion matrix:")
    print(result["test_cm"])
print("\nCompleted runs:", len(run_results))



# per-site validation performance

site_predictions_df = pd.concat(
    site_predictions,
    ignore_index=True
)

site_run_results = []
for (site, seed), group in site_predictions_df.groupby( # calculate metrics separately for each site and seed
    ["SITE_ID", "Seed"]):
    actual = group["Actual"].to_numpy()
    predicted = group["Predicted"].to_numpy()
    probabilities = group["Probability"].to_numpy()
    accuracy = accuracy_score(actual, predicted)
    macro_f1 = f1_score(
        actual,
        predicted,
        average="macro",
        zero_division=0
    )
    if len(np.unique(actual)) == 2: # since AUC is undefined if a site only has 1 class
        auc = roc_auc_score(actual, probabilities)
    else:
        auc = np.nan
    site_run_results.append({
        "Model": "GCN",
        "Seed": seed,
        "SITE_ID": site,
        "N": len(group),
        "Controls": int(np.sum(actual == 0)),
        "ASD": int(np.sum(actual == 1)),
        "Accuracy": accuracy,
        "Macro_F1": macro_f1,
        "ROC_AUC": auc
    })

site_run_results = pd.DataFrame(site_run_results)

print("\n" + "=" * 60)
print("GCN PER-SITE VALIDATION RESULTS — EACH RUN")
print("=" * 60)

print(
    site_run_results.sort_values(
        ["SITE_ID", "Seed"]
    ).to_string(index=False)
)


# mean and sd across 3 runs for each site
site_summary = (
    site_run_results
    .groupby(["Model", "SITE_ID"], as_index=False)
    .agg(
        N=("N", "first"),
        Controls=("Controls", "first"),
        ASD=("ASD", "first"),

        Mean_Accuracy=("Accuracy", "mean"),
        SD_Accuracy=("Accuracy", "std"),

        Mean_Macro_F1=("Macro_F1", "mean"),
        SD_Macro_F1=("Macro_F1", "std"),

        Mean_ROC_AUC=("ROC_AUC", "mean"),
        SD_ROC_AUC=("ROC_AUC", "std")
    )
)

print("\n" + "=" * 60)
print("GCN per-site Validation Results - mean +/- sd")
print("=" * 60)

print(
    site_summary.sort_values("SITE_ID").to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# saving both detailed and summary tables
output_dir = "/Users/harshithabhupathiraju/ABIDE_Project"

site_run_results.to_csv(
    f"{output_dir}/gcn_per_site_validation_each_run.csv",
    index=False
)
site_summary.to_csv(
    f"{output_dir}/gcn_per_site_validation_summary.csv",
    index=False
)
print("Per-site CSV files saved successfully.")

