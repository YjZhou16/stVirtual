# Quick start

1. Follow [Installation](installation.md) and activate the environment.
2. Obtain the data listed in [Data Availability](Data-Availability.md).
3. Choose a workflow from the tutorial navigation and open its experiment configuration.
4. Follow the preprocessing, representation, decoder, and transport steps in the order shown in that tutorial.
5. Inspect simulation results and use the corresponding perturbation workflow to explore interventions.

## Experiment paths

Run commands from the repository root. Paths in an experiment configuration are resolved relative to that experiment directory.

For Mcardiac_EA, place the raw Slide-seq files under `experiments/Mcardiac_EA/data/GSE282547` and the coordinate files under `experiments/Mcardiac_EA/data/coordinates`. These directories can also be symbolic links to your existing data directories.

## Explore a complete workflow

[3D tissue reconstruction](wiki/01-3d-tissue-reconstruction.md) covers the spatial RNA and measured-protein examples. [Developmental reconstruction across time](wiki/02-developmental-reconstruction-across-time.md) follows Membryo and Mcardiac_EA. [Cancer progression across stages](wiki/03-cancer-progression-across-stages.md) covers human gastric and lung cancer. [4D spatiotemporal development reconstruction](wiki/05-4d-spatiotemporal-development-reconstruction.md) follows Mcardiac. The [notebook examples](notebooks.md) provide the corresponding code and visualizations.
