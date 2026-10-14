from nilearn import datasets

print("Downloading ABIDE dataset...")

abide = datasets.fetch_abide_pcp(
    data_dir="/Users/harshithabhupathiraju/ABIDE_Project/abide_data",
    pipeline="cpac",
    derivatives=["rois_cc200"],
    quality_checked=True
)

print("Download complete!")

print("Number of subjects:", len(abide.rois_cc200))
print("Phenotypic file:", abide.phenotypic)