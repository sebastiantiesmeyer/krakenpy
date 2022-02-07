from turtle import update
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from numpy.lib.arraysetops import unique
from numpy.lib.type_check import _nan_to_num_dispatcher

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors



class SpatialData():

    def __init__(self, gene_annotations, x_coordinates, y_coordinates):#, data_frame, gene_tag='gene', x_tag='X', y_tag='Y'):

        self.data = pd.DataFrame({'gene_annotations':gene_annotations,'X':x_coordinates,'Y':y_coordinates})
        self.raw = self.data.copy()

        self.uns = {}

        self.stats = None
        self.update_stats()

    @property
    def gene_annotations(self):
        return self.data['gene_annotations']

    @property
    def X(self):
        return self.data['X']

    @property
    def Y(self):
        return self.data['Y']

    @property
    def counts(self):
        return self.stats['counts']

    @property
    def counts_sorted(self):
        return self.stats.counts[self.count_idcs]

    @property
    def gene_classes_sorted(self):
        return self.gene_classes[self.count_idcs]

    @property
    def gene_classes(self):
        return self.stats['gene_classes']

    @property
    def count_ranks(self):
        return self.stats['count_ranks']

    def update_stats(self): 

        gene_classes, indicers, inverse, counts = np.unique(
            self.gene_annotations,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )

        self.count_idcs=np.argsort(counts)
        count_ranks = np.argsort(self.count_idcs)
        
        # print(inverse, indicers)

        self.stats = pd.DataFrame({'gene_classes':gene_classes,
                                    'counts':counts,
                                    'count_ranks':count_ranks
                                    },
                                    index=gene_classes)

        self.data['gene_id'] = inverse

    def get_count(self, gene):
        if gene in self.gene_classes.values:
            return int(self.stats.counts[self.gene_classes == gene])

    def get_count_rank(self, gene):
        if gene in self.gene_classes.values:
            return int(self.count_ranks[self.gene_classes == gene])

    def filter(self, genes_to_keep):
        spatial_filter = self.gene_annotations.isin(genes_to_keep)

        self.data=self.data[spatial_filter]
        self.update_stats()

    def knn_entropy(self, n_neighbors=4):

        knn = NearestNeighbors(n_neighbors=n_neighbors)

        coordinates = np.stack([self.X, self.Y]).T
        knn.fit(coordinates)
        distances, indices = knn.kneighbors(coordinates)
        # return (indices)

        knn_cells = np.zeros_like(indices)
        for i in range(indices.shape[1]):
            knn_cells[:, i] = self.data['gene_id'].iloc[indices[:, i]]

        H = np.zeros((len(self.gene_classes), ))

        for i, gene in enumerate(self.gene_classes):
            x = knn_cells[self.data['gene_id'] == i]
            _, n_x = np.unique(x[:,1:], return_counts=True)
            p_x = n_x / n_x.sum()
            h_x = -(p_x * np.log2(p_x)).sum()
            H[i] = h_x

        return (H)

    def plot_entropy(self, n_neighbors=4):

        H = self.knn_entropy(n_neighbors)

        idcs = np.argsort(H)
        plt.figure(figsize=(25, 25))

        fig, axd = plt.subplot_mosaic([
            ['scatter_1', 'scatter_2', 'scatter_3', 'scatter_4'],
            ['bar', 'bar', 'bar', 'bar'],
            ['scatter_5', 'scatter_6', 'scatter_7', 'scatter_8'],
        ],
                                      figsize=(11, 7),
                                      constrained_layout=True)

        dem_plots = np.array([
            0,
            2,
            len(H) - 3,
            len(H) - 1,
            1,
            int(len(H) / 2),
            int(len(H) / 2) + 1,
            len(H) - 2,
        ])
        colors = ('royalblue', 'goldenrod', 'red', 'purple', 'lime',
                  'turquoise', 'green', 'yellow')

        axd['bar'].bar(
            range(len(H)),
            H[idcs],
            color=[
                colors[np.where(
                    dem_plots == i)[0][0]] if i in dem_plots else 'grey'
                for i in range(len(self.stats.counts))
            ])

        axd['bar'].set_xticks(range(len(H)),
                              [self.gene_classes[h] for h in idcs],
                              rotation=90)
        axd['bar'].set_ylabel('knn entropy, k=' + str(n_neighbors))

        for i in range(8):
            idx = idcs[dem_plots[i]]
            gene = self.gene_classes[idx]
            plot_name = 'scatter_' + str(i + 1)
            axd[plot_name].set_title(gene)
            axd[plot_name].scatter(self.X, self.Y, color=(0.5, 0.5, 0.5, 0.1))
            axd[plot_name].scatter(self.X[self.data['gene_id'] == idx],
                                   self.Y[self.data['gene_id'] == idx],
                                   color=colors[i],
                                   marker='.')
            axd[plot_name].set_xticks([], [])
            # if i>0:
            axd[plot_name].set_yticks([], [])

            if i < 4:
                y_ = (H[idcs])[i]
                _y = 0
            else:
                y_ = 0
                _y = 1

            con = ConnectionPatch(xyA=(dem_plots[i], y_),
                                  coordsA=axd['bar'].transData,
                                  xyB=(np.mean(axd[plot_name].get_xlim()),
                                       axd[plot_name].get_ylim()[_y]),
                                  coordsB=axd[plot_name].transData,
                                  color='white',
                                  linewidth=1,
                                  linestyle='dotted')
            fig.add_artist(con)

    def scatter(self, gene, **kwargs):

        axd = plt.subplot(111)

        axd.set_title(gene)
        axd.scatter(self.X, self.Y, color=(0.5, 0.5, 0.5, 0.1))
        axd.scatter(self.X[self.gene_annotations == gene], self.Y[self.gene_annotations == gene],
                    **kwargs)

    def plot_bars(self, axis=None, **kwargs):
        if axis is None:
            axis = plt.subplot(111)
        axis.bar(np.arange(len(self.stats.counts)), self.counts_sorted, **kwargs)
        axis.set_yscale('log')

        axis.set_xticks(
            np.arange(len(self.gene_classes_sorted)),
            self.gene_classes_sorted,
            # fontsize=12,
            rotation=90)

        axis.set_ylabel('molecule count')

    def plot_overview(self):
        plt.style.use('dark_background')
        colors = ('royalblue', 'goldenrod', 'red', 'lime')

        scatter_idcs = np.round(np.linspace(0,
                                            len(self.stats.counts) - 1,
                                            4)).astype(int)

        fig, axd = plt.subplot_mosaic(
            [['scatter_1', 'scatter_2', 'scatter_3', 'scatter_4'],
             ['bar', 'bar', 'bar', 'bar']],
            figsize=(11, 7),
            constrained_layout=True)

        self.plot_bars(
            axd['bar'],
            color=[
                colors[np.where(
                    scatter_idcs == i)[0][0]] if i in scatter_idcs else 'grey'
                for i in range(len(self.stats.counts))
            ])

        # bbox = axd['bar'].get_window_extent().transformed(
        #     fig.dpi_scale_trans.inverted())
        # width = bbox.width

        # print(width, fig.dpi)

        for i in range(4):
            idx = self.count_idcs[scatter_idcs[i]]
            gene = self.gene_classes[idx]
            plot_name = 'scatter_' + str(i + 1)
            axd[plot_name].set_title(gene)
            axd[plot_name].scatter(self.X, self.Y, color=(0.5, 0.5, 0.5, 0.1))
            axd[plot_name].scatter(self.X[self.data['gene_id']  == idx],
                                   self.Y[self.data['gene_id']  == idx],
                                   color=colors[i],
                                   marker='.')
            axd[plot_name].set_xticks([], [])
            # if i>0:
            axd[plot_name].set_yticks([], [])

            con = ConnectionPatch(xyA=(scatter_idcs[i], self.stats.counts[idx]),
                                  coordsA=axd['bar'].transData,
                                  xyB=(np.mean(axd[plot_name].get_xlim()),
                                       axd[plot_name].get_ylim()[0]),
                                  coordsB=axd[plot_name].transData,
                                  color='white',
                                  linewidth=1,
                                  linestyle='dotted')
            fig.add_artist(con)

        plt.suptitle('Selected Expression Densities:', fontsize=18)


def determine_gains(sc, spatial):

    sc_genes = sc.var.index
    counts_sc = np.array(sc.X.sum(0) / sc.X.sum()).flatten()
    counts_spatial = np.array([spatial.get_count(g) for g in sc_genes])
    counts_spatial = counts_spatial / counts_spatial.sum()
    count_ratios = counts_sc / counts_spatial
    return count_ratios


def plot_gains(sc, spatial):

    sc_genes = sc.var.index

    # counts_spatial_reindexed = counts_spatial [[spatial.get_sort_index(g) for g in sc.unique_genes_sorted]]
    count_ratios = determine_gains(sc, spatial)
    idcs = np.argsort(count_ratios)

    ax = plt.subplot(111)

    span = np.linspace(0, 1, len(idcs))
    clrs = np.stack([
        span,
        span * 0,
        span[::-1],
    ]).T

    ax.barh(range(len(count_ratios)), np.log(count_ratios[idcs]), color=clrs)

    ax.text(0,
            len(idcs) + 3,
            'lost in spatial ->',
            ha='center',
            fontsize=12,
            color='red')
    ax.text(0, -3, '<- lost in SC', ha='center', fontsize=12, color='lime')

    for i, gene in enumerate(sc_genes[idcs]):
        if count_ratios[idcs[i]] < 1:
            ha = 'left'
            xpos = 0.05
        else:
            ha = 'right'
            xpos = -0.05
        ax.text(0, i, gene, ha=ha)

    ax.set_yticks([], [])


def compare_counts(sc, spatial):

    sc_genes = (sc.var.index)
    sc_counts = (np.array(sc.X.sum(0)).flatten())
    sc_count_idcs = np.argsort(sc_counts)
    count_ratios = np.log(determine_gains(sc, spatial))
    count_ratios -= count_ratios.min()
    count_ratios /= count_ratios.max()

    ax1 = plt.subplot(311)
    ax1.set_title('compared molecule counts:')
    ax1.bar(np.arange(len(sc_counts)),sc_counts[sc_count_idcs], color='grey')
    ax1.set_ylabel('log(count) scRNAseq')
    ax1.set_xticks(np.arange(len(sc_genes)),sc_genes[sc_count_idcs],  rotation=90)
    ax1.set_yscale('log')

    ax2 = plt.subplot(312)
    for i, gene in enumerate(sc_genes[sc_count_idcs]):
        plt.plot(
            [i, spatial.get_count_rank(gene)],
            [1, 0],
        )
    plt.axis('off')
    ax2.set_ylabel(' ')

    ax3 = plt.subplot(313)
    spatial.plot_bars(ax3, color='grey')
    ax3.invert_yaxis()
    ax3.xaxis.tick_top()
    ax3.xaxis.set_label_position('top')
    ax3.set_ylabel('log(count) spatial')

def get_count(adata,gene):
    return(adata.X[:,adata.var.index==gene].sum())

def compare_counts_cellwise(sc, spatial, cell_obs_label):

    genes = list(sc.unique_genes)
    count_ratios = np.log(determine_gains(sc, spatial))
    count_ratios -= count_ratios.min()
    count_ratios /= count_ratios.max()

    ax1 = plt.subplot(311)

    # print([genes.index(g) for g in sc.unique_genes])

    span = np.array([count_ratios[sc.get_index(g)] for g in sc.unique_genes_sorted])
    clrs = np.stack([
        span,
        span * 0,
        1 - span,
    ]).T

    ax1.set_title('compared molecule counts:')
    sc.plot_bars(ax1, color=clrs)
    ax1.set_ylabel('log(count) scRNAseq')


    ax2 = plt.subplot(312)
    for i, gene in enumerate(sc.unique_genes_sorted):
        plt.plot(
            [i, spatial.get_sort_index(gene)],
            [1, 0],
        )
    plt.axis('off')
    ax2.set_ylabel(' ')

    ax3 = plt.subplot(313)

    span = np.array([count_ratios[sc.get_index(g)] for g in spatial.unique_genes_sorted])

    clrs = np.stack([
        span,
        span * 0,
        1 - span,
    ]).T

    spatial.plot_bars(ax3, color=clrs)
    ax3.invert_yaxis()
    ax3.xaxis.tick_top()
    ax3.xaxis.set_label_position('top')
    ax3.set_ylabel('log(count) spatial')

def plot_overview(spatial):
    spatial.plot_overview()

def synchronize(sc, spatial, mRNA_threshold_sc=10, mRNA_threshold_spatial=10):

    genes_sc = set(sc.var.index)
    genes_spatial = set(spatial.gene_classes)

    genes_xor = genes_sc.symmetric_difference(genes_spatial)

    for gene in genes_xor:
        if gene in genes_sc:
            pass
            # print('Removing {}, which is not present in spatial data'.format(
            #     gene))
        else:
            print('Removing {}, which is not present in SC data'.format(gene))

    genes_and = (genes_sc & genes_spatial)

    for gene in sorted(genes_and):
        print(gene)
        if  (get_count(sc,gene) < mRNA_threshold_sc):
            print('Removing {}, low count in SC data'.format(gene))
            genes_and.remove(gene)
        elif spatial.get_count(gene) < mRNA_threshold_spatial:
            print('Removing {}, low count in spatial data'.format(gene))
            genes_and.remove(gene)

    genes_and = sorted(genes_and)

    spatial.filter(genes_and)
    sc = sc[:,genes_and]


    return(sc,spatial)
