# Datasets

The dataset files in this directory are extracted from the supplied `data.zip`. Original directory names and file contents are preserved. Operating-system metadata is omitted.

| Path | Dataset |
| --- | --- |
| `IMDB/` | IMDB arrays |
| `mnistm/` | MNIST-M train and test Parquet files |
| `TinyImageNet/` | TinyImageNet pickle files |
| `N24news/` | N24News train and test JSONL files |
| `NoisyAG-News.pkl` | NoisyAG-News |
| `HR/` | HR CSV files |
| `House/` | House CSV files |
| `bike/` | Bike CSV files |

The previously included [20 Newsgroups](../mydatasets-dl/news/), [MNIST and Fashion-MNIST](../mydatasets-dl/), and [Adult](../mydatasets/adult/) files remain in their original repository directories. N24News and 20 Newsgroups are separate datasets.

## Obtaining the Files

The N24News JSONL files are stored with Git LFS. With Git LFS installed, clone the repository and retrieve its data:

```bash
git lfs install
git clone https://github.com/maoyuecong682/ICDE2027_EVA.git
cd ICDE2027_EVA
git lfs pull
```

For an existing clone, run `git pull` followed by `git lfs pull`.

[manifest.json](manifest.json) records the source archive SHA256 and each dataset file's size and SHA256. The archive contributes 69 dataset files; existing datasets outside `data/` are retained.
