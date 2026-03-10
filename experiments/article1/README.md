# Article 1 Experiments

Dataset inspection and visualization scripts.

## Scripts

- `inspect_awf_dataset.py` - Visualize AWF dataset
- `inspect_turbopark_dataset.py` - Visualize TurbOPark dataset
- `plot_dataset_distributions.py` - Plot dataset distributions
- `plot_inflow_layout_generator.py` - Generate layout/inflow figures

## Usage

Run from the `data-generation/` directory:

```bash
python experiments/article1/inspect_awf_dataset.py --data-dir data/awf_10layouts_test
python experiments/article1/inspect_turbopark_dataset.py --data-dir data/turbopark_10layouts_test
python experiments/article1/plot_inflow_layout_generator.py
```

## Directory Structure

- `cache/` - Cached data (git-ignored)
- `figures/` - Generated figures (git-ignored)
- `outputs/` - Inspection outputs (git-ignored)
