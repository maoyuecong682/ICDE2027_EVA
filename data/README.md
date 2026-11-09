# Original EVA Datasets

The bundled files are copied byte-for-byte from the original EVA directories on the server. IMDB-Large, TinyImageNet, MNIST-M, and CIFAR-10 are download-only; none of their dataset files are bundled.

| Repository directory | Files | Size (MiB) |
| --- | ---: | ---: |
| `data/` | 42 | 23.63 |
| `mydatasets/adult/` | 2 | 5.23 |
| `mydatasets-dl/` | 18,863 | 193.30 |

`data/` contains HR, House, and bike. `mydatasets/adult/` contains the original Adult train/test CSVs. `mydatasets-dl/` contains Fashion-MNIST, MNIST, and 20 Newsgroups. Counts and sizes exclude this README and the manifest.

## Downloads

| Dataset | Download | Local destination |
| --- | --- | --- |
| IMDB-Large (`IMDBLargeC5`) | [Prepared classification archive](https://drive.google.com/file/d/1IglePB_S5l4iDCoEHxdwRn11yuWcUptF/view), provided by [RECON](https://github.com/for0nething/RECON/blob/main/README.md) | `data/IMDB/` |
| TinyImageNet | [Official image archive](https://cs231n.stanford.edu/tiny-imagenet-200.zip) | `data/TinyImageNet/` after preprocessing |
| MNIST-M | [Train Parquet](https://huggingface.co/datasets/Mike0307/MNIST-M/resolve/bca0c963fd95756a36b8fbcff13417770310ffb6/data/train-00000-of-00001-571b6b1e2c195186.parquet), [test Parquet](https://huggingface.co/datasets/Mike0307/MNIST-M/resolve/bca0c963fd95756a36b8fbcff13417770310ffb6/data/test-00000-of-00001-ba3ad971b105ff65.parquet); [original author release](https://yaroslav.ganin.net/) | `data/mnistm/mnistm_train.parquet`, `data/mnistm/mnistm_test.parquet` |
| CIFAR-10 | [Official Python archive](https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz), [dataset page](https://cave.cs.toronto.edu/kriz/cifar.html) | Extract under the data path passed to `torchvision.datasets.CIFAR10` |

IMDB-Large uses the prepared `IMDBLargeC5` arrays from the linked RECON release. The Stanford movie-review sentiment data used by `fetch_imdb.py` and `rebuild_imdb_large.py` is a separate dataset.

The original TinyImageNet loader expects `(data, labels)` in `train.pkl`, `valid.pkl`, and `test.pkl`; the official ZIP contains images and requires preprocessing. The existing `eva_output/build_tinyimagenet_pkl.py` instead consumes Parquet files from the [zh-plus mirror](https://huggingface.co/datasets/zh-plus/tiny-imagenet) and writes JPEG-byte pickles for `tinyimagenet_modified.py` at its configured paths.

Save the MNIST-M Parquet downloads using the destination filenames above. Their `image` and `label` columns match the loader format. The first 64 KiB of each download matches the corresponding original server file; full-file equivalence could not be checked against the truncated originals.

## File Manifest

[manifest.json](manifest.json) lists the size and SHA256 of all 18,907 bundled dataset files, plus the omitted paths and download sources. All 52 files belonging to the four download-only datasets are excluded, including the 25 files previously flagged as incomplete. Checks cover file structure, not full semantic validation.

These directories are separate from the later prepared baseline dataset bundle used through `EVA_DATA_ROOT`.
