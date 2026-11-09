# Original EVA Datasets

These files are copied from the original EVA directories on the server. Original directory names and file bytes are preserved.

| Repository directory | Files | Size (MiB) |
| --- | ---: | ---: |
| `data/` | 92 | 172.07 |
| `mydatasets/adult/` | 2 | 5.23 |
| `mydatasets-dl/` | 18,865 | 222.54 |

`data/` contains HR, House, IMDB, TinyImageNet, bike, and MNIST-M files. `mydatasets/adult/` contains the original Adult train/test CSVs. `mydatasets-dl/` contains downloaded CIFAR-10, Fashion-MNIST, MNIST, and 20 Newsgroups files.

## Integrity

**25 original files are incomplete or truncated.** They are included as historical originals and must be replaced before they can be used for training. The affected files are IMDB arrays, TinyImageNet pickle files, MNIST-M Parquet files, and two CIFAR-10 archives.

See [manifest.json](manifest.json) for every file's size, SHA256, and integrity status. A file without a flagged structural issue is not a guarantee of complete semantic validation.

These directories are separate from the later prepared baseline dataset bundle used through `EVA_DATA_ROOT`. The original case-sensitive names, such as `IMDB` and `TinyImageNet`, are retained.
