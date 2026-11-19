# NeuroGuideNet

## Dataset

### BCI Competition IV-2a dataset

Download **BCI Competition IV-2a dataset (BCIC-IV-2A)** from the official BBCI website.

1. **EEG recordings (`.gdf`)** — Open the [official download page](https://bbci.de/competition/iv/download/) and select **GDF files zipped** under **Data sets 2a**. Extract the recordings into `BCICIV_2a_gdf/`.
2. **Evaluation labels (`.mat`)** — On the [official results page](https://www.bbci.de/competition/iv/results/#labels), select **labels** next to **Data sets 2a** to download [`true_labels.zip`](https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip). Extract the label files into `BCICIV_2a_gdf/true_labels/`.

The directory tree below shows the files for subject `A01`.

```text
BCICIV_2a_gdf/
├── A01T.gdf          # Training-session recording
├── A01E.gdf          # Evaluation-session recording
└── true_labels/
    ├── A01T.mat      # Training-session labels
    └── A01E.mat      # Evaluation-session labels
```

The same layout applies to all nine subjects, `A01` through `A09`. Each
`.gdf` recording has a matching `.mat` label file with the same base name
in `true_labels/`.

### OpenBMI dataset

