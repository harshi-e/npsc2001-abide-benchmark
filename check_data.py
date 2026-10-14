import numpy as np
from nilearn import datasets

abide = datasets.fetch_abide_pcp(
    data_dir="/Users/harshithabhupathiraju/ABIDE_Project/abide_data",
    pipeline="cpac",
    derivatives=["rois_cc200"],
    quality_checked=True
)

ts = abide.rois_cc200[0] # brain-region time series of first subject

print("Type:")
print(type(ts))
print("\nTime series shape:")
print(ts.shape)
print("\nNumber of subjects:")
print(len(abide.rois_cc200))
print("\nFirst 5 rows and 5 brain regions:")
print(ts[:5, :5])

corr_matrix = np.corrcoef(ts, rowvar=False) # functional connectivity matrix
print("\nCorrelation matrix shape:")
print(corr_matrix.shape)