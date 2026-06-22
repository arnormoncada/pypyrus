# SDK Quickstart

This walkthrough uses the [plant seedlings](https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset) example as a small end-to-end PyPyrus demo: clone the repo, fetch the dataset, run one training job, then inspect what PyPyrus recorded.

## Setup

The plant seedlings data already lives in `experiments/plant_seedlings/data/` in this repo. It is stored with Git LFS, so fetch it after cloning:

```bash
git clone https://github.com/arnormoncada/pypyrus.git
cd pypyrus

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

git lfs install
git lfs pull

pip install -e .
```

If you have an older machine like me, and usually run into `torch` /
`torchvision` / `numpy` compatibility issues, use the pinned installer
script which installs the versions I used for development 🧌:

```bash
./scripts/install_old_mac_env.sh
```

## Run The Demo

Run the example with the built-in split and the demo run name:

```bash
python experiments/plant_seedlings/train_mobilenetv3_small.py \
  --data-root experiments/plant_seedlings/data/split \
  --epochs 2 \
  --num-workers 0 \
  --run-name plant-seedlings-demo
```

This is a real training run, so expect it to take a few minutes on CPU. It was also usde for the experiments, so it contains extra configuration and timing code.

## Inspect The Run

PyPyrus writes to `./pypyrus.db` by default, so you can inspect the run directly with the CLI.

List runs:

```bash
pypyrus runs list
```

Find the row named `plant-seedlings-demo` and copy its `run_id`.

Show the run:

```bash
pypyrus runs show <run_id>
```

Look for:

- `Run name: plant-seedlings-demo`
- two `ImageFolder` datasets
- `train` and `test` roles
- `sample_id_scheme: filepath`
- `sample_id_resolver: file_collection`

Show the first recorded batch:

```bash
pypyrus batches show <run_id> --step 0
```

`--step` is run-global. In this output, the sample IDs should look like `filepath:<class>/<filename>`.

Find one sample from that batch:

```bash
pypyrus samples find <run_id> --sample-id <sample_id_from_step_0>
```

Paste one sample ID from the `batches show` output and confirm that PyPyrus reports where that sample appeared.

## What Is PyPyrus Code Here?

Most of [train_mobilenetv3_small.py](../experiments/plant_seedlings/train_mobilenetv3_small.py) is ordinary PyTorch or Python. The PyPyrus-specific part is small:

```python
with Run(..., run_name=args.run_name) as run:
    train_loader = attach(train_loader, run, role="train")
    test_loader = attach(test_loader, run, role="test")
```

- `Run` creates the tracked run and stores the human-readable run name. In this walkthrough, `args.run_name` is `plant-seedlings-demo`.
- `attach(..., role="train")` instruments the training loader.
- `attach(..., role="test")` instruments the evaluation loader.

## Why Sample IDs Work Automatically

`torchvision.datasets.ImageFolder` already fits PyPyrus's built-in file-collection contract:

- `dataset.root` points at the split root such as `.../data/split/train`
- `dataset.samples` is a list of entries shaped like `(path, class_index)`

That lets PyPyrus store sample IDs as `filepath:<relative-path>`, for example `filepath:Charlock/383.png`.

Because this walkthrough uses separate `train/` and `test/` roots, PyPyrus registers two dataset identities: one for the training split and one for the test split.

## Another Example: Forest Covertype

PyPyrus also includes [train_covtype_mlp.py](../experiments/forest_covertype/train_covtype_mlp.py), a tabular CSV example based on the [forest covertype dataset](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset).

That example shows the more explicit PyPyrus path:

```python
def covtype_sample_id_resolver(dataset, index, sample):
    row = dataset.rows[index]
    return SampleIdResolution(
        sample_id=f"record_id:{row['sample_id']}",
        sample_id_scheme="record_id",
        sample_id_resolver="user_override",
    )


with Run(..., run_name=args.run_name) as run:
    train_loader = attach(
        train_loader,
        run,
        role="train",
        sample_id_resolver=covtype_sample_id_resolver,
        dataset_uri=str(data_path),
    )
```

- `dataset_uri=str(data_path)` tells PyPyrus exactly which CSV file the dataset came from. Strongest way to control dataset identity.
- `covtype_sample_id_resolver(...)` reads the stable row ID from `dataset.rows[index]` and stores sample IDs like `record_id:12345`.
- This shows that how you can seamlessly integrate PyPyrus with your own dataset types, even if they don't fit the built-in file-collection or structured-record contracts.

You can run the covertype example with:

```bash
python experiments/forest_covertype/train_covtype_mlp.py \
  --data-path experiments/forest_covertype/data/covtype_with_sample_id.csv \
  --epochs 2 \
  --num-workers 0 \
  --run-name forest-covertype-demo
```
