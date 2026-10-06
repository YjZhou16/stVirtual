"""Consistent annotation colors and before/after coordinate plots."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def annotation_palette(adata, key, palette_path):
    palette = json.loads(Path(palette_path).read_text())
    labels = adata.obs[key].astype(str)
    if hasattr(adata.obs[key], 'cat') and key + '_colors' in adata.uns:
        stored = dict(zip(map(str, adata.obs[key].cat.categories), adata.uns[key + '_colors']))
        for label, color in stored.items():
            palette.setdefault(label, color)
    missing = sorted(set(labels) - set(palette))
    colors = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('tab20b').colors) + list(plt.get_cmap('tab20c').colors)
    for i, label in enumerate(missing):
        palette[label] = colors[i % len(colors)]
    return palette

def _legend(fig, labels, palette, columns=4):
    if max(map(len, labels), default=0) > 32:
        columns = min(columns, 2)
    handles = [Line2D([], [], marker='o', linestyle='', color=palette[x], markersize=5, label=x) for x in labels]
    nrows = int(np.ceil(len(labels) / columns))
    bottom = min(.42, (.23 * nrows + .35) / fig.get_figheight())
    fig.legend(handles=handles, loc='lower center', ncol=columns, frameon=False, fontsize=8)
    fig.subplots_adjust(bottom=bottom, top=.90, wspace=.25, hspace=.28)

def _scatter(ax, coords, labels, palette, *, dimension, flip_y, size, alpha=.7):
    coords = np.asarray(coords).copy()
    if flip_y: coords[:, 1] *= -1
    ax.scatter(*coords[:, :dimension].T, c=[palette[x] for x in labels], s=size, alpha=alpha,
               linewidths=0, rasterized=True)
    if dimension == 3:
        ax.view_init(elev=22, azim=42)
        ax.set_box_aspect(np.maximum(np.ptp(coords[:, :3], axis=0), 1e-6))
        ax.set_zlabel('Z')
    else:
        ax.set_aspect('equal')
    ax.set_xlabel('X');ax.set_ylabel('Y')

def sample_panels(adata, coordinates, sample_key, label_key, palette, *, title, dimension=2, flip_y=True, size=1):
    samples = adata.obs[sample_key].astype(str).to_numpy()
    labels = adata.obs[label_key].astype(str).to_numpy()
    groups = list(pd.unique(samples));cols = min(3, len(groups));rows = int(np.ceil(len(groups)/cols))
    fig=plt.figure(figsize=(5*cols,4.5*rows+1.5),dpi=130)
    for i,group in enumerate(groups):
        ax=fig.add_subplot(rows,cols,i+1,projection='3d' if dimension==3 else None)
        m=samples==group
        _scatter(ax,np.asarray(coordinates)[m],labels[m],palette,dimension=dimension,flip_y=flip_y,size=size)
        ax.set_title(group)
    fig.suptitle(title);_legend(fig,sorted(set(labels)),palette)
    plt.show()
    plt.close(fig)
    return fig

def alignment_overlay(adata, before, after, sample_key, label_key, palette, *, dimension=2, flip_y=True, size=1, by_sample=False):
    key=sample_key if by_sample else label_key
    labels=adata.obs[key].astype(str).to_numpy();cats=list(pd.unique(labels))
    if by_sample:palette=dict(zip(cats,plt.get_cmap('tab10').colors))
    fig=plt.figure(figsize=(13,6.5),dpi=130)
    for i,(coords,title) in enumerate([(before,'Before alignment'),(after,'After alignment')]):
        ax=fig.add_subplot(1,2,i+1,projection='3d' if dimension==3 else None)
        _scatter(ax,coords,labels,palette,dimension=dimension,flip_y=flip_y,size=size,alpha=.35 if by_sample else .65)
        ax.set_title(title)
    _legend(fig,sorted(set(labels)),palette)
    plt.show()
    plt.close(fig)
    return fig

def embedding_panels(adata, color_keys, palette_path, *, basis='X_umap', size=3):
    coords=np.asarray(adata.obsm[basis])[:,:2]
    for key in color_keys:
        labels=adata.obs[key].astype(str).to_numpy()
        palette=annotation_palette(adata,key,palette_path)
        columns = 2 if max(map(len, labels), default=0) > 32 else 3
        height = max(7, 5 + .26 * int(np.ceil(len(set(labels)) / columns)))
        fig,ax=plt.subplots(figsize=(9,height),dpi=130)
        ax.scatter(*coords.T,c=[palette[x] for x in labels],s=size,lw=0,rasterized=True)
        ax.set_title(key);ax.set_axis_off();_legend(fig,sorted(set(labels)),palette,columns=3)
        plt.show()
        plt.close(fig)
