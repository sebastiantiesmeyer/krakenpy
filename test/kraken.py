from gc import collect
from math import dist
from sqlite3 import SQLITE_CREATE_TABLE
from turtle import back, distance, update
from types import CellType
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from numpy.lib.arraysetops import unique
from numpy.lib.type_check import _nan_to_num_dispatcher
from scipy import sparse
import collections

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from skimage.transform import rescale  #from scikit-image import , resize, downscale_local_mean


class PixelMap():

    def __init__(self,
                 pixel_data,
                 name=None,
                 upscale=1,
                 pixel_data_original=None):

        self.data = pixel_data
        if pixel_data_original is not None:
            self.pixeld_data_original = pixel_data_original
        else:
            self.pixeld_data_original = pixel_data

        self.n_channels = 1 if len(
            pixel_data.shape) == 2 else pixel_data.shape[-1]

        if not isinstance(upscale, collections.Iterable) or len(upscale) == 1:
            self.scale = (upscale, upscale)
        else:
            self.scale = upscale

        self.extent = (0, pixel_data.shape[0] / self.scale[0], 0,
                       pixel_data.shape[1] / self.scale[1])

    @property
    def shape(self):
        return self.extent[1], self.extent[3]

    # def rescale(self, factor):

    #     if not isinstance(factor, collections.Iterable) or len(factor)==1:
    #         factor = (factor,factor)

    #     origin = (self.extent[0],self.extent[2])
    #     self.extent = (origin[0], origin[0] + self.pixel_data.shape[1] * factor[0],
    #                    origin[1], origin[1] + self.pixel_data.shape[0] * factor[1])

    def imshow(self, **kwargs):

        extent = np.array(self.extent)
        # print(extent)

        plt.imshow(self.data**0.2, extent=extent[[0, 3, 1, 2]], **kwargs)

    def __getitem__(self, indices):
        # print(indices)

        if not isinstance(indices, collections.Iterable):
            index_x = indices
            index_y = slice(0, None, None)

        else:
            index_x = indices[0]

            if len(indices) > 1:
                index_y = indices[1]
            else:
                index_y = slice(0, None, None)

        if (index_x.start is None): start_x = 0  #self.extent[0]
        else: start_x = index_x.start
        if (index_x.stop is None): stop_x = self.extent[1]
        else: stop_x = index_x.stop

        if (index_y.start is None): start_y = 0  #self.extent[2]
        else: start_y = index_y.start
        if (index_y.stop is None): stop_y = self.extent[3]
        else: stop_y = index_y.stop

        data = self.data[int(start_y * self.scale[1]):int(stop_y *
                                                          self.scale[1]),
                         int(start_x * self.scale[0]):int(stop_x *
                                                          self.scale[0]), ]

        return PixelMap(
            data,
            upscale=self.scale,
        )


class SpatialIndexer():

    def __init__(self, df):
        self.df = df

    @property
    def shape(self):
        if self.df.background is None:
            return np.ceil(self.df.X.max() - self.df.X.min()).astype(
                int), np.ceil(self.df.Y.max() - self.df.Y.min()).astype(int)

    def create_cropping_mask(self, start, stop, series):

        # print(start, stop)
        if start is None:
            start = 0

        if stop is None:
            stop = series.max()

        return ((series > start) & (series < stop))

    def join_cropping_mask(self, xlims, ylims):
        return self.create_cropping_mask(
            *xlims, self.df.X) & self.create_cropping_mask(*ylims, self.df.Y)

    def crop(self, xlims, ylims):

        mask = self.join_cropping_mask(xlims, ylims)

        pixel_maps = []

        if xlims[0] is None:
            start_x = 0
        else:
            start_x = xlims[0]
        if ylims[0] is None:
            start_y = 0
        else:
            start_y = ylims[0]

        for pm in self.df.pixel_maps:
            pixel_maps.append(pm[xlims[0]:xlims[1], ylims[0]:ylims[1]])

        return SpatialData(self.df.gene_annotations[mask],
                           self.df.X[mask] - start_x,
                           self.df.Y[mask] - start_y, pixel_maps)

    def __getitem__(self, indices):

        if not isinstance(indices, collections.Iterable):
            indices = (indices, )
        if len(indices) == 1:
            ylims = (0, None)
        else:
            ylims = (indices[1].start, indices[1].stop)

        xlims = (indices[0].start, indices[0].stop)

        return self.crop(xlims, ylims)


class SpatialData(pd.DataFrame):

    def __init__(self,
                 gene_annotations,
                 x_coordinates,
                 y_coordinates,
                 pixel_maps=[]):

        super(SpatialData, self).__init__({
            'gene_annotations': gene_annotations,
            'X': x_coordinates,
            'Y': y_coordinates
        })

        self._metadata = ['uns', 'stats', 'pixel_maps']
        self.uns = {'background': None}
        self.pixel_maps = []

        for pm in pixel_maps:
            if not type(pm) == PixelMap:
                self.pixel_maps.append(PixelMap(pm))
            else:
                self.pixel_maps.append(pm)

        self.update_stats()

    # @property
    # def count_idcs(self):
    #     return self.stats.count_indices

    @property
    def gene_ids(self):
        return self.gene_id

    @property
    def counts(self):
        return self.stats['counts']

    @property
    def counts_sorted(self):
        return self.stats.counts[self.stats.count_indices]

    @property
    def gene_classes_sorted(self):
        return self.gene_classes[self.stats.count_indices]

    @property
    def gene_classes(self):
        return self.stats.index

    # @property
    # def count_ranks(self):
    #     return self.stats['count_ranks']

    @property
    def spatial(self):
        return SpatialIndexer(self)

    @property
    def background(self):
        if len(self.pixel_maps):
            return self.pixel_maps[0]

        #.__getitem__(self, arg):

    def __getitem__(self, *arg):

        if (len(arg) == 1):

            if type(arg[0]) == str:

                return super().__getitem__(arg[0])

            if (type(arg[0]) == slice):
                new_data = super().iloc.__getitem__(arg)

            elif (type(arg[0]) == int):
                new_data = super().iloc.__getitem__(slice(arg[0], arg[0] + 1))

            elif isinstance(arg[0], pd.Series):
                # print(arg[0].values)
                new_data = super().iloc.__getitem__(arg[0].values)

            elif isinstance(arg[0], np.ndarray):
                new_data = super().iloc.__getitem__(arg[0])

            elif isinstance(arg[0], collections.Sequence):
                if all([a in self.keys() for a in arg[0]]):
                    return super().__getitem__(*arg)
                new_data = super().iloc.__getitem__(arg[0])

            new_frame = SpatialData(new_data.gene_annotations, new_data.X,
                                    new_data.Y, self.pixel_maps)
            new_frame.update_stats()
            return (new_frame)

        print('Converting to generic Pandas.')
        return super().__getitem__(*arg)

    def update_stats(self):

        gene_classes, indicers, inverse, counts = np.unique(
            super().__getitem__(['gene_annotations']),
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )

        count_idcs = np.argsort(counts)
        count_ranks = np.argsort(count_idcs)

        self.stats = pd.DataFrame(
            {
                # 'gene_classes': gene_classes,
                'counts': counts,
                'count_ranks': count_ranks,
                'count_indices': count_idcs,
                'gene_ids': np.arange(len(gene_classes))
            },
            index=gene_classes)

        self['gene_id'] = inverse

        if 'knn_distances' in self.uns:
            del self.uns['knn_distances']
        if 'knn_indices' in self.uns:
            del self.uns['knn_indices']

    def get_count(self, gene):
        if gene in self.gene_classes.values:
            return int(self.stats.counts[self.gene_classes == gene])

    def get_id(self, gene_name):
        return int(self.stats.gene_ids[self.gene_classes == gene_name])

    def get_count_rank(self, gene):
        if gene in self.gene_classes.values:
            return int(self.stats.count_ranks[self.gene_classes == gene])

    def knn(self, n_neighbors=4, re_run=False):

        if not all(
            (k in self.uns for k in ['knn_distances', 'knn_indices'])) or (
                n_neighbors > self.uns['knn_distances'].shape[1]) or re_run:
            knn = NearestNeighbors(n_neighbors=n_neighbors)
            coordinates = np.stack([self.X, self.Y]).T
            knn.fit(coordinates)
            distances, indices = knn.kneighbors(coordinates)
            self.uns['knn_distances'] = distances
            self.uns['knn_indices'] = indices
            self.uns['knn_types'] = np.array(self.gene_ids)[indices]

        return self.uns['knn_distances'], self.uns['knn_indices'], self.uns[
            'knn_types']

    def knn_entropy(self, n_neighbors=4):

        _, indices, _ = self.knn(n_neighbors=n_neighbors)

        knn_cells = np.zeros_like(indices)
        for i in range(indices.shape[1]):
            knn_cells[:, i] = self['gene_id'].iloc[indices[:, i]]

        H = np.zeros((len(self.gene_classes), ))

        for i, gene in enumerate(self.gene_classes):
            x = knn_cells[self['gene_id'] == i]
            _, n_x = np.unique(x[:, 1:], return_counts=True)
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
            axd[plot_name].scatter(self.X[self['gene_id'] == idx],
                                   self.Y[self['gene_id'] == idx],
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

    def scatter(self,
                c=None,
                color=None,
                gene=None,
                axd=None,
                plot_bg=True,
                **kwargs):

        if axd is None:
            axd = plt.subplot(111)

        if self.background and plot_bg:
            self.background.imshow(cmap='Greys', alpha=0.3)

        if c is None and color is None:
            c = self.gene_ids

        # axd.set_title(gene)
        axd.scatter(self.X,
                    self.Y,
                    c=c,
                    color=color,
                    cmap='nipy_spectral',
                    **kwargs)

    def plot_bars(self, axis=None, **kwargs):
        if axis is None:
            axis = plt.subplot(111)
        axis.bar(np.arange(len(self.stats.counts)), self.counts_sorted,
                 **kwargs)
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

        for i in range(4):
            idx = self.stats.count_indices[scatter_idcs[i]]
            gene = self.gene_classes[idx]
            plot_name = 'scatter_' + str(i + 1)
            axd[plot_name].set_title(gene)
            axd[plot_name].scatter(self.X, self.Y, color=(0.5, 0.5, 0.5, 0.1))
            axd[plot_name].scatter(self.X[self['gene_id'] == idx],
                                   self.Y[self['gene_id'] == idx],
                                   color=colors[i],
                                   marker='.')
                                   
            axd[plot_name].set_xticks([], [])
            # if i>0:
            axd[plot_name].set_yticks([], [])

            con = ConnectionPatch(xyA=(scatter_idcs[i],
                                       self.stats.counts[idx]),
                                  coordsA=axd['bar'].transData,
                                  xyB=(np.mean(axd[plot_name].get_xlim()),
                                       axd[plot_name].get_ylim()[0]),
                                  coordsB=axd[plot_name].transData,
                                  color='white',
                                  linewidth=1,
                                  linestyle='dotted')
            fig.add_artist(con)

        plt.suptitle('Selected Expression Densities:', fontsize=18)

    def plot_radial_distribution(self, n_neighbors=30, **kwargs):
        distances, _, _ = self.knn(n_neighbors=n_neighbors)
        plt.hist(distances[:, 1:n_neighbors].flatten(), **kwargs)

    def spatial_decomposition(
        self,
        mRNAs_center=None,
        mRNAs_neighbor=None,
        n_neighbors=10,
    ):

        if mRNAs_center is None:
            mRNAs_center = self.gene_classes
        if mRNAs_neighbor is None:
            mRNAs_neighbor = self.gene_classes

        _, neighbors, _ = self.knn(n_neighbors=n_neighbors)
        neighbor_classes = np.array(self.gene_ids)[neighbors]

        pptx = []
        ppty = []
        clrs = []
        intensity = []

        # ids_neighbor = np.array([self.get_id(m) for m in mRNAs_neighbor])

        out = np.zeros((100, 100, len(self.gene_classes)))

        mask_center = np.logical_or.reduce(
            [neighbor_classes[:, 0] == self.get_id(m) for m in mRNAs_center])

        for i_n_neighbor in range(1, n_neighbors):

            mask_neighbor = np.logical_or.reduce([
                neighbor_classes[:, i_n_neighbor] == self.get_id(m)
                for m in mRNAs_neighbor
            ])

            mask_combined = np.logical_and(mask_center, mask_neighbor)
            # distances_filtered = distances[mask_combined]

            for i_neighbor, n in enumerate(neighbors[mask_combined]):

                xs = np.array(self.X.iloc[n])
                ys = np.array(self.Y.iloc[n])

                x_centered = xs - xs[0]
                y_centered = ys - ys[0]

                loc_neighbor = np.array(
                    (x_centered[i_n_neighbor], y_centered[i_n_neighbor]))
                loc_neighbor_normalized = loc_neighbor / (loc_neighbor**
                                                          2).sum()**0.5

                rotation_matrix = np.array(
                    [[loc_neighbor_normalized[1], -loc_neighbor_normalized[0]],
                     [loc_neighbor_normalized[0], loc_neighbor_normalized[1]]])

                rotated_spots = np.inner(
                    np.array([x_centered, y_centered]).T, rotation_matrix).T

                # we want to exclude the central and n_neighbor spots:
                mask = np.arange(rotated_spots.shape[1])
                mask = (mask > 1) & (mask != (i_n_neighbor))

                #         #         plt.scatter(rotated[0][mask],rotated[1][mask])

                pptx.append(rotated_spots[0][mask])
                ppty.append(rotated_spots[1][mask])
                clrs.append(self.gene_ids.iloc[n][mask])
                # intensity.append(1 / distances_filtered[i_neighbor])

                mask = (pptx[-1] >= -50) & (pptx[-1] < 50) & (
                    ppty[-1] >= -50) & (ppty[-1] < 50)
                out[ppty[-1].astype(int)[mask], pptx[-1].astype(int)[mask],
                    clrs[-1][mask]] += 1  #intensity[-1][mask]

        #         break

        pptx = np.concatenate(pptx)
        ppty = np.concatenate(ppty)
        clrs = np.concatenate(clrs)

        plt.axhline(0)
        plt.axvline(0)

        for u in np.unique(clrs):
            plt.scatter(pptx[clrs == u], ppty[clrs == u], alpha=0.1)
        plt.scatter(
            0,
            0,
            color='lime',
            s=70,
        )

        # intensity = np.concatenate(intensity)
        return np.roll(np.roll(out[1:, 1:], 50, axis=1), 50, axis=0)

    def knn_clean(
        self,
        n_neighbors=10,
    ):
        distances, indices, types = self.knn(n_neighbors=n_neighbors)
        count_matrix = sparse.lil_matrix(
            (types.shape[0], self.gene_classes.shape[0]))
        for i, t in enumerate(types):
            classes, counts = (np.unique(t[:n_neighbors], return_counts=True))
            count_matrix[i, classes] = counts / counts.sum()

        count_matrix = count_matrix.tocsr()

        count_matrix_log = count_matrix.copy()
        count_matrix_log.data = np.log(count_matrix.data)
        count_matrix_inv = count_matrix.copy()
        count_matrix_inv.data = 1 / (count_matrix.data)

        prototypes = np.zeros((len(self.gene_classes), ) * 2)
        for i in range(prototypes.shape[0]):
            prototypes[i] = count_matrix[self.gene_ids == i].sum(0)
        prototypes /= prototypes.sum(0)

        Dkl = count_matrix.copy()

        for i in range(prototypes.shape[0]):
            inter = Dkl[self.gene_ids == i]
            inter.data = count_matrix[self.gene_ids == i].data * (np.log(
                (count_matrix_inv[self.gene_ids == i].multiply(
                    prototypes[i])).data))
            Dkl[self.gene_ids == i] = inter
        Dkl = -np.array(Dkl.sum(1)).flatten()
        Dkl[np.isinf(Dkl)] = 0

        return Dkl


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
    ax1.bar(np.arange(len(sc_counts)), sc_counts[sc_count_idcs], color='grey')
    ax1.set_ylabel('log(count) scRNAseq')
    ax1.set_xticks(np.arange(len(sc_genes)),
                   sc_genes[sc_count_idcs],
                   rotation=90)
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


def get_count(adata, gene):
    return (adata.X[:, adata.var.index == gene].sum())


def compare_counts_cellwise(sc, spatial, cell_obs_label):

    genes = list(sc.unique_genes)
    count_ratios = np.log(determine_gains(sc, spatial))
    count_ratios -= count_ratios.min()
    count_ratios /= count_ratios.max()

    ax1 = plt.subplot(311)

    # print([genes.index(g) for g in sc.unique_genes])

    span = np.array(
        [count_ratios[sc.get_index(g)] for g in sc.unique_genes_sorted])
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

    span = np.array(
        [count_ratios[sc.get_index(g)] for g in spatial.unique_genes_sorted])

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
        if (get_count(sc, gene) < mRNA_threshold_sc):
            print('Removing {}, low count in SC data'.format(gene))
            genes_and.remove(gene)
        elif spatial.get_count(gene) < mRNA_threshold_spatial:
            print('Removing {}, low count in spatial data'.format(gene))
            genes_and.remove(gene)

    genes_and = sorted(genes_and)

    # spatial.filter(genes_and)
    sc = sc[:, genes_and]

    return (sc, spatial[spatial.gene_annotations.isin(genes_and)])
