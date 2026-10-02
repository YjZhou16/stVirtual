# stVirtual tutorials

stVirtual is a niche-driven multi-agent generative framework for reconstructing spatiotemporal 3D and 4D tissue dynamics from sparse and static measurements across space, time, and disease progression.

The simulation below illustrates mouse cardiac development from E9.5 to E11.5. stVirtual reconstructs intermediate three-dimensional tissue states across developmental time, showing changes in tissue organization and the spatial distribution of cell populations.

![Mcardiac cardiac development from E9.5 to E11.5](assets/results/mcardiac.gif)

Start with installation and a complete experiment, then explore the workflows for spatial transport, transition dynamics, and perturbation.

| Workflow | Data and examples |
| --- | --- |
| [3D tissue reconstruction](wiki/01-3d-tissue-reconstruction.md) | HMLN, Mbrain, and Human breast cancer |
| [Developmental reconstruction across time](wiki/02-developmental-reconstruction-across-time.md) | Membryo and Mcardiac_EA |
| [Cancer progression across stages](wiki/03-cancer-progression-across-stages.md) | Human gastric cancer and Human lung cancer |
| [4D tissue reconstruction](wiki/04-4d-tissue-reconstruction.md) | Workflow template |
| [4D spatiotemporal development reconstruction](wiki/05-4d-spatiotemporal-development-reconstruction.md) | Mcardiac |
| [Perturbation](wiki/06-perturbation.md) | Cell removal, expression suppression, and cell-state transition or signaling interventions |

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
Data-Availability
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

wiki/01-3d-tissue-reconstruction
wiki/02-developmental-reconstruction-across-time
wiki/03-cancer-progression-across-stages
wiki/04-4d-tissue-reconstruction
wiki/05-4d-spatiotemporal-development-reconstruction
wiki/06-perturbation
```

```{toctree}
:maxdepth: 2
:caption: Examples

notebooks
citation
```
