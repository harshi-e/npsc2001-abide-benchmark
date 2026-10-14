# npsc2001-abide-benchmark
Benchmarking neuroimaging models for autism classification under distribution shift
# Benchmarking Autism Classification Models Under Distribution Shift

# Project Overview

This project benchmarks 3 machine learning approaches for classifying Autism Spectrum Disorder (ASD) versus control participants using functional connectivity data from the Autism Brain Imaging Data Exchange (ABIDE).

The project compares:

- Logistic Regression (LR): A conventional statistical classification baseline implemented in R.
- Multilayer Perceptron (MLP): A feedforward neural network implemented in R.
- Graph Convolutional Network (GCN): A graph-based neural network implemented in Python.

The models are evaluated using accuracy, macro-averaged F1-score and ROC-AUC. Model performance is assessed across three random seeds, with a particular focus on generalisation to an unseen acquisition site.

# Dataset

The project uses ABIDE preprocessed functional MRI data from the CPAC pipeline, with the `rois_cc200.1D` regional time-series files.

- Participants: 871
- Brain regions: 200
- Classification task: ASD versus control
- Labels: 'DX_GROUP = 1` for ASD and `DX_GROUP = 0` for controls
- Site identifier: `SITE_ID`

Functional connectivity is represented using correlations between regional time series.

The GCN represents each participant as a graph with 200 nodes, connectivity-based node features and edges constructed using an absolute connectivity threshold of 0.2.

Ensure that dataset access and redistribution comply with the applicable ABIDE data-use requirements. The raw dataset is not included in this repository unless explicitly permitted.

# Data Splitting

A fixed site-based split is used to evaluate generalisation to an unseen acquisition site.

Partition               Participants
Training                559
Validation              140
Held-out NYU test set   172
Total                   871

All NYU participants are excluded from training and validation and reserved for the final held-out evaluation.

The split indices are saved in `graphs/split_indices.pt` and reused to maintain consistent participant partitions.
Three random seeds are used to assess variability between runs. Mean and standard deviation are reported for the evaluation metrics.

# Project Structure

- `README.Rmd` — Project overview and instructions.
- `graphs/` — Processed graph dataset and split indices.
- `download_abide.py` — Dataset download or retrieval.
- `check_data.py` — Data checks.
- `build_one_graph.py` — Construction of an individual participant graph.
- `build_dataset.py` — Construction of the graph dataset.
- `split_dataset.py` — Dataset partitioning.
- `export_split.py` — Export or saving of split indices.
- `gcn_model.py` — GCN training and evaluation.
- `lr_mlp_models` — R scripts for Logistic Regression and MLP.
- `results/` — Model metrics, confusion matrices and other evaluation outputs.

# Requirements

# Python
The GCN implementation uses Python and PyTorch Geometric.
The environment should include the packages required by the scripts, including:

- PyTorch
- PyTorch Geometric
- NumPy
- pandas
- scikit-learn

Install the project's recorded Python dependencies using the instructions below, once a `requirements.txt` file has been created and verified.

python -m pip install -r requirements.txt

PyTorch and PyTorch Geometric installation instructions may depend on the operating system and available hardware. Use compatible versions for the selected environment.

# R

The LR and MLP implementations use R and the packages required by the analysis scripts. These include `caret`, `pROC` and `nnet`.

Install any missing packages before running the R analyses. Record the actual package dependencies and versions used for the final experiments.

# Running the Project

# 1. Prepare the data

Run the data retrieval and checking scripts as required:

python download_abide.py
python check_data.py

If the dataset is already downloaded and verified, data retrieval may be skipped.

# 2. Build the graph dataset

python build_dataset.py

This step should create the graph dataset required by the GCN. If the script does not automatically construct individual graphs, run the graph-building utility in the order required by the implementation.

# 3. Create and export the splits

Run the split-generation and export scripts as required:

python split_dataset.py
python export_split.py

Verify that the saved split indices correspond to the intended training, validation and held-out NYU partitions.
Do not regenerate the splits unnecessarily when reproducing the reported experiments.

# 4. Train and evaluate the GCN

python gcn_model.py

The script trains the GCN using the training partition, selects a checkpoint using validation loss, and evaluates performance on the validation set and held-out NYU test set.
The implementation uses three random seeds: 123, 456 and 789.

# 5. Run Logistic Regression and MLP

Both conventional baseline models are implemented in the same R script, `NPSC ASSIGNMENT`.

Open the script in RStudio or run it from the command line:

Rscript "lr_mlp_models.R"

The script performs the Logistic Regression and MLP analyses, including preprocessing, model evaluation and calculation of the required performance metrics.
Ensure that the input dataset and participant split information are available at the paths expected by the script before running the analyses.

# Evaluation Metrics

The models are evaluated using:

- Accuracy: Proportion of correctly classified participants.
- Macro-F1: Unweighted average of the F1-scores for ASD and control classes.
- ROC-AUC: Ability to rank ASD participants above control participants.
- Confusion matrix: Counts of correct and incorrect predictions for each class.

Results are summarised across three random seeds using the mean and standard deviation.

The NYU test partition provides the primary evaluation of generalisation to an unseen acquisition site. Per-site validation results are reported separately and interpreted cautiously because some sites contain very few validation participants.

# Reproducibility

To reproduce the experiments:

1. Set up the required Python and R environments.
2. Obtain the ABIDE data and verify the required ROI files.
3. Generate or retrieve the processed graph dataset.
4. Reuse the saved participant split indices.
5. Run the LR, MLP and GCN evaluation scripts.
6. Save the resulting metrics, confusion matrices and timing information.
7. Compare the outputs with the reported results.

Random seeds, data splits, preprocessing steps and model configurations should remain consistent with those documented in the report.

# Limitations

Performance on an unseen acquisition site may differ from performance on the validation set because of differences in acquisition protocols, participant characteristics and other sources of distribution shift.

Site-level validation metrics are exploratory where sample sizes are small. Results from three seeds quantify variability across those runs but do not establish statistical significance or robustness across all possible acquisition sites.
