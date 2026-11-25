# NeuroGuideNet

## Dataset

### BCI Competition IV-2a dataset

Download **BCI Competition IV-2a dataset (BCIC-IV-2A)** from the official BBCI website.

1. **EEG recordings (`.gdf`)** — Open the [official download page](https://bbci.de/competition/iv/download/) and select **GDF files zipped** under **Data sets 2a**. Extract the recordings into `BCICIV_2a_gdf/`.
2. **Evaluation labels (`.mat`)** — On the [official results page](https://www.bbci.de/competition/iv/results/#labels), select **labels** next to **Data sets 2a** to download [`true_labels`](https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip). Move the resulting `true_labels` folder into `BCICIV_2a_gdf/` so the label files are in `BCICIV_2a_gdf/true_labels/`.

The directory tree below shows the files for subject `A01`.

```text
BCICIV_2a_gdf/
├── A01T.gdf          # Training-session recording
├── A01E.gdf          # Evaluation-session recording
├── ...
└── true_labels/
    ├── A01T.mat      # Training-session labels
    ├── A01E.mat      # Evaluation-session labels
    └── ...
```

The same layout applies to all nine subjects, `A01` through `A09`. Each
`.gdf` recording has a matching `.mat` label file with the same base name
in `true_labels/`.

Set `dataset.root` in [the BCI IV-2a configuration](config/BCIIV2a_NeuroGuideNet.yaml) to your local `BCICIV_2a_gdf/` directory.

### OpenBMI dataset

NeuroGuideNet uses the motor imagery (MI) recordings from the [official OpenBMI dataset on GigaDB (DOI 10.5524/100542)](https://doi.org/10.5524/100542).

GigaDB organizes the recordings in session and subject folders, for example `session1/s1/sess01_subj01_EEG_MI.mat`.

The [data loader](dataset/OpenBMI_data_mat.py) reads each `sessNN_subjNN_EEG_MI.mat` file directly from `dataset.root`. The local directory below shows the files for subject 1 in both sessions.

```text
OpenBMI/
├── sess01_subj01_EEG_MI.mat
├── sess02_subj01_EEG_MI.mat
└── ...
```

- **Manual download** — Download only the `*_EEG_MI.mat` files for the subjects and sessions you plan to use. The other files in the GigaDB record, including ERP and SSVEP recordings, can be skipped. Keep the original filenames and place the files directly in `OpenBMI/`.
- **Scripted download** — Run [the OpenBMI download script](dataset/OpenBMI_download.py) from the repository root. It saves the MI files for both sessions directly in the directory specified by `--save-dir`.

Set `dataset.root` in [the OpenBMI configuration](config/OpenBMI_NeuroGuideNet.yaml) to your local `OpenBMI/` directory.

