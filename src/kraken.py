
from __future__ import annotations
from cgitb import text
from enum import unique

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

from typing import Union

from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.cm import get_cmap
import matplotlib.patheffects as PathEffects


from scipy import sparse
import collections
import scanpy as sc
import anndata
import scipy

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

from umap import UMAP
from sklearn.manifold import TSNE


class PixelMap():

    def __init__(self: PixelMap,
                 pixel_data: np.ndarray,
                 upscale: float = 1.0,) -> None:

        self.data = pixel_data

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
        return self.extent[1] - self.extent[0], self.extent[3] - self.extent[2]

    def imshow(self, axd=None, **kwargs) -> None:
        extent = np.array(self.extent)

        if (len(self.data.shape)>2) and (self.data.shape[2]>4):
            data = self.data.sum(-1)
        else:
            data = self.data

        if axd is None:
            axd = plt.subplot(111)

        axd.imshow(data, extent=extent[[0, 3, 1, 2]], **kwargs)

    def __getitem__(self, indices: Union[slice, collections.Iterable[slice]]):

        if not isinstance(indices, collections.Iterable):
            index_x = indices
            index_y = slice(0, None, None)

        else:
            index_x = indices[0]

            if len(indices) > 1:
                index_y = indices[1]
            else:
                index_y = slice(0, None, None)

        if (index_x.start is None):
            start_x = 0  # self.extent[0]
        else:
            start_x = index_x.start
        if (index_x.stop is None):
            stop_x = self.extent[1]
        else:
            stop_x = index_x.stop

        if (index_y.start is None):
            start_y = 0  # self.extent[2]
        else:
            start_y = index_y.start
        if (index_y.stop is None):
            stop_y = self.extent[3]
        else:
            stop_y = index_y.stop

        data = self.data[int(start_y * self.scale[1]):int(stop_y *
                                                          self.scale[1]),
                         int(start_x * self.scale[0]):int(stop_x *
                                                          self.scale[0]), ]

        return PixelMap(
            data,
            upscale=self.scale,
        )

class KDEProjection(PixelMap):
    def __init__(self,sd: SpatialData,
                 bandwidth: float = 3.0,
                 threshold_vf_norm: float = 1.0,
                 threshold_p_corr: float = 0.5,
                 upscale: float = 1) -> None:
        
        self.sd = sd
        self.bandwidth = bandwidth
        self.threshold_vf_norm = threshold_vf_norm
        self.threshold_p_corr = threshold_p_corr

        self.scale = upscale

        super().__init__(self.run_kde(), upscale)



    def run_kde(self) -> None:

        kernel = self.generate_kernel(self.bandwidth*3, self.scale)

        x_int = np.array(self.sd.y * self.scale).astype(int)
        y_int = np.array(self.sd.x * self.scale).astype(int)
        genes = self.sd.gene_ids

        vf = np.zeros((x_int.max()+kernel.shape[0]+1,y_int.max()+kernel.shape[0]+1,len(self.sd.genes)))

        for x,y,g in zip(x_int,y_int,genes):
            # print(x,y,vf.shape,kernel.shape)
            vf[x:x+kernel.shape[0],y:y+kernel.shape[1],g]+=kernel
            
        return vf[kernel.shape[0]//2:-kernel.shape[0]//2,kernel.shape[1]//2:-kernel.shape[1]//2]

    def generate_kernel(self, bandwidth: float, scale: float = 1) -> np.ndarray:

        kernel_width_in_pixels = int(bandwidth * scale *
                                     6)  # kernel is 3 sigmas wide.

        span = np.linspace(-3, 3, kernel_width_in_pixels)
        X, Y = np.meshgrid(span, span)

        return 1 / (2 * np.pi)**0.5 * np.exp(-0.5 * ((X**2 + Y**2)**0.5)**2)


class CellTypeMap(PixelMap):
    def __init__(self,data,celltype_labels,*args,**kwargs):
        
        # .super().

        pass
    # def __init__(self: PixelMap, pixel_data: np.ndarray, upscale: float = 1) -> None:
    #     super().__init__(pixel_data, upscale)

# class SsamLite(PixelMap):

#     def __init__(self,
#                  sd: SpatialData,
#                  bandwidth: float = 3.0,
#                  threshold_vf_norm: float = 1.0,
#                  threshold_p_corr: float = 0.5,
#                  upscale: float = 1) -> None:

#         self.sd = sd
#         self.bandwidth = bandwidth
#         self.threshold_vf_norm = threshold_vf_norm
#         self.threshold_p_corr = threshold_p_corr

#         self.extent = tuple(self.sd.background.extent)
#         self.extent_shape = (int(self.extent[1] - self.extent[0]),
#                              int(self.extent[3] - self.extent[2]))

#         self.scale = upscale
#         self.data = np.zeros(
#             (int(self.shape[0] * self.scale), int(self.shape[1] * self.scale)))


#     @property
#     def signatures(self) -> np.ndarray:
#         return self.sd.scanpy.signatures

#     def run_algorithm(self) -> None:

#         # from tqdm import tqdm_notebook

#         kernel = self.generate_kernel(15, self.scale)

#         X_int = np.array(self.sd.y * self.scale).astype(int)
#         Y_int = np.array(self.sd.x * self.scale).astype(int)
#         sort_idcs = np.argsort(X_int)
#         X_ = X_int[sort_idcs]
#         Y_ = Y_int[sort_idcs]

#         uniques, inv_idcs = np.unique(X_, return_index=True)

#         gene_ids = np.array(self.sd.gene_ids)[sort_idcs]

#         print(self.celltype_map.shape)

#         temp_vf = np.zeros((kernel.shape[0],
#                             self.celltype_map.shape[1] + kernel.shape[0] + 10,
#                             len(self.sd.genes)))

#         for i, u_ in enumerate(uniques[:-1]):

#             _u = uniques[i + 1]

#             u_u = min(_u - u_, kernel.shape[0])

#             y = Y_[inv_idcs[i]:inv_idcs[i + 1]]
#             ids = gene_ids[inv_idcs[i]:inv_idcs[i + 1]]

#             for i, k in enumerate(kernel):
#                 temp_vf[:, y + i, ids] += k[:, None]

#             self.celltype_map[u_:u_ + u_u] = temp_vf[-u_u:, 19:].sum(-1)

#             temp_vf[-u_u:] = 0

#             temp_vf = np.roll(temp_vf, u_u, axis=0)

#     def generate_kernel(self, bandwidth: float, scale: float = 1) -> np.ndarray:

#         kernel_width_in_pixels = int(bandwidth * scale *
#                                      6)  # kernel is 3 sigmas wide.

#         span = np.linspace(-3, 3, kernel_width_in_pixels)
#         X, Y = np.meshgrid(span, span)

#         return 1 / (2 * np.pi)**0.5 * np.exp(-0.5 * ((X**2 + Y**2)**0.5)**2)

class SpatialGraph():

    def __init__(self, df, n_neighbors=10) -> None:

        self.df = df
        self.n_neighbors = n_neighbors
        self._neighbors = None
        self._neighbor_types = None
        self._distances = None
        self._umap = None
        self._tsne = None

    @property
    def neighbors(self):
        if self._neighbors is None:
            self._distances, self._neighbors, self._neighbor_types = self.update_knn(
                self.n_neighbors)
        return self._neighbors[:,:self.n_neighbors]

    @property
    def distances(self):
        if self._distances is None:
            self._distances, self._neighbors, self._neighbor_types = self.update_knn(
                self.n_neighbors)
        return self._distances[:,:self.n_neighbors]

    @property
    def neighbor_types(self):
        if self._neighbor_types is None:
            self._distances, self._neighbors, self._neighbor_types = self.update_knn(
                self.n_neighbors)
        return self._neighbor_types[:,:self.n_neighbors]

    @property
    def umap(self):
        if self._umap is None:
            self.run_umap()
        return self._umap

    @property
    def tsne(self):
        if self._tsne is None:
            self.run_tsne()
        return self._tsne

    def __getitem__(self,*args):
        sg = SpatialGraph(self.df,self.n_neighbors)
        if self._distances is not None:
            sg._distances = self._distances.__getitem__(*args)
        if self._neighbors is not None:
            sg._neighbors = self._neighbors.__getitem__(*args)
        if self._neighbor_types is not None:
            sg._neighbor_types = self._neighbor_types.__getitem__(*args)

    def update_knn(self, n_neighbors, re_run=False):

        if self._neighbors is not None and (n_neighbors <
                                            self._neighbors.shape[1]):
            self.n_neighbors=n_neighbors
            # return (self._neighbors[:, :n_neighbors],
            #         self._distances[:, :n_neighbors],
            #         self._neighbor_types[:, :n_neighbors])
        else:

            coordinates = np.stack([self.df.x, self.df.y]).T
            knn = NearestNeighbors(n_neighbors=n_neighbors)
            knn.fit(coordinates)
            self._distances, self._neighbors = knn.kneighbors(coordinates)
            self._neighbor_types = np.array(self.df.gene_ids)[self._neighbors]

            self.n_neighbors = n_neighbors

            # return self.distances, self.neighbors, self.neighbor_types

    def knn_entropy(self, n_neighbors=4):

        self.update_knn(n_neighbors=n_neighbors)
        indices = self.neighbors  # (n_neighbors=n_neighbors)

        knn_cells = np.zeros_like(indices)
        for i in range(indices.shape[1]):
            knn_cells[:, i] = self.df['gene_id'].iloc[indices[:, i]]

        H = np.zeros((len(self.df.genes), ))

        for i, gene in enumerate(self.df.genes):
            x = knn_cells[self.df['gene_id'] == i]
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
                for i in range(len(self.df.stats.counts))
            ])

        axd['bar'].set_xticks(range(len(H)),
                              [self.df.genes[h] for h in idcs],
                              rotation=90)
        axd['bar'].set_ylabel('knn entropy, k=' + str(n_neighbors))

        for i in range(8):
            idx = idcs[dem_plots[i]]
            gene = self.df.genes[idx]
            plot_name = 'scatter_' + str(i + 1)
            axd[plot_name].set_title(gene)
            axd[plot_name].scatter(self.df.x, self.df.y, color=(0.5, 0.5, 0.5, 0.1))
            axd[plot_name].scatter(self.df.x[self.df['gene_id'] == idx],
                                   self.df.y[self.df['gene_id'] == idx],
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

    def _determine_counts(self,bandwidth=1):
        counts = np.zeros((len(self.df,),len(self.df.genes)))

        for i in range(self.df.graph.n_neighbors):
            counts[np.arange(len(self.df)),self.df.graph.neighbor_types[:,i]]+=np.exp(-self.df.graph.distances[:,i]**2/(2*bandwidth**2)) 
        return counts

    def run_umap(self,bandwidth=1,*args,**kwargs):        

        counts = self._determine_counts(bandwidth=bandwidth)
        umap=UMAP(*args,**kwargs)
        self._umap = umap.fit_transform(counts)

    def run_tsne(self,bandwidth=1,*args,**kwargs):        
        counts = self._determine_counts(bandwidth=bandwidth)
        tsne=TSNE(*args,**kwargs)
        self._tsne = tsne.fit_transform(counts)

    def plot_umap(self, text_column=None, color_category='g', color_dict=None, c=None, **kwargs):
        self.plot_embedding(self.umap, text_column=text_column, color_category=color_category, color_dict=color_dict, c=c, **kwargs)

    def plot_tsne(self,text_column=None, color_category='g', color_dict=None, c=None, **kwargs):
        self.plot_embedding(self.tsne, text_column=text_column, color_category=color_category, color_dict=color_dict, c=c, **kwargs)

    def plot_embedding(self, embedding, text_column=None, color_category='g', color_dict=None, c=None, **kwargs):

        categories = self.df[color_category].unique() 

        if (color_dict is None):
            cmap = get_cmap('nipy_spectral')
            color_dict = {categories[i]:cmap(f) for i,f in enumerate(np.linspace(0,1,len(categories)))}

        colors = [color_dict[c] for c in self.df[color_category]]
        handlers = [plt.scatter([],[],color=color_dict[c]) for c in color_dict]

        if c is not None:
            colors=c
        plt.legend(handlers, color_dict.keys())

        plt.scatter(*embedding.T,c=colors, **kwargs)

        if text_column is not None:
            cog_dict = self._determine_cog(embedding,text_column)
            unique_texts = list(cog_dict.keys())
            cogs = np.array(list(cog_dict.values()))
            cogs = self._untangle_text(cogs,min_distance=0.9)

            for i,g in enumerate(unique_texts):
                x = cogs[i][0]
                y = cogs[i][1]
                txt = plt.text(x,y,g,size=13,ha='center',weight='bold',color='w')
                # txt = plt.text(x,y,g,size=13,ha='center',color=color ,weight='bold')

                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='k')])

    

    def _determine_cog(self, embedding, column_name):

        unique_texts = self.df[column_name].unique()
        codes = self.df[column_name].cat.codes

        knn = NearestNeighbors(n_neighbors=50)
        knn.fit(embedding)
        distances,neighbors = knn.kneighbors(embedding)
        neighbor_types = np.array(codes)[neighbors]
        cogs = []

        for i in range(len(unique_texts)):
            nt_filtered = 1/(1+distances)
            nt_filtered[(neighbor_types[:,0]!=i)]=0
            nt_filtered[(neighbor_types!=i)]=0
            gravity = nt_filtered
            gravity = gravity.sum(1)
            cog = gravity.argmax()
            cogs.append(embedding[cog])

        return {self.df[column_name].cat.categories[i]: c for i,c in enumerate(cogs)}
 
    def _untangle_text(self, cogs, untangle_rounds=50, min_distance=0.5):
        knn = NearestNeighbors(n_neighbors=2)

        cogs_new = cogs.copy()

        for i in range(untangle_rounds):

            cogs = cogs_new.copy()
            knn = NearestNeighbors(n_neighbors=2)

            knn.fit(cogs)
            distances,neighbors = knn.kneighbors(cogs)
            too_close = (distances[:,1]<min_distance)

            for i,c in enumerate(np.where(too_close)[0]):
                partner = neighbors[c,1]
                cog = cogs[c]-cogs[partner]
                cog_new = cogs[c]+0.3*cog
                cogs_new[c]= cog_new
                
        return cogs_new


class ScanpyDataFrame():

    def __init__(self, sd, scanpy_ds):
        self.sd = sd
        self.adata = scanpy_ds
        self.stats = ScStatistics(self)
        self.celltype_labels = None
        self.signature_matrix = None

    @property
    def shape(self):
        return self.adata.shape

    def generate_signatures(self, celltype_obs_marker='celltype'):

        self.celltype_labels = np.unique(self.adata.obs[celltype_obs_marker])

        self.signature_matrix = np.zeros((
            len(self.celltype_labels),
            self.adata.shape[1],
        ))

        for i, label in enumerate(self.celltype_labels):
            self.signature_matrix[i] = np.array(
                self.adata[self.adata.obs[celltype_obs_marker] == label].X.sum(
                    0)).flatten()

        self.signature_matrix = self.signature_matrix - self.signature_matrix.mean(
            1)[:, None]
        self.signature_matrix = self.signature_matrix / self.signature_matrix.std(
            1)[:, None]

        self.signature_matrix = pd.DataFrame(self.signature_matrix, index=self.celltype_labels,columns=self.stats.index )

        return self.signature_matrix

    def synchronize(self):

        joined_genes = (self.stats.genes.intersection(self.sd.genes)).sort_values()

        # print(len(joined_genes))

        self.sd.reset_index()
        self.adata = self.adata[:, joined_genes]
        self.stats = ScStatistics(self)
        self.sd.drop(index=list(
            self.sd.index[~self.sd.g.isin(joined_genes)]),
            inplace=True)

        self.sd.stats = PointStatistics(self.sd)

        self.sd.graph = SpatialGraph(self.sd)


    def determine_gains(self):

        sc_genes = self.adata.var.index
        counts_sc = np.array(self.adata.X.sum(0) / self.adata.X.sum()).flatten()
        
        counts_spatial = np.array([self.sd.stats.get_count(g) for g in sc_genes])

        counts_spatial = counts_spatial / counts_spatial.sum()
        count_ratios = counts_sc / counts_spatial
        return count_ratios


    def plot_gains(self):

        sc_genes = self.adata.var.index

        count_ratios = self.determine_gains()
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


    def compare_counts(self):

        sc_genes = (self.adata.var.index)
        sc_counts = (np.array(self.adata.X.sum(0)).flatten())
        sc_count_idcs = np.argsort(sc_counts)
        count_ratios = np.log(self.determine_gains())
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
                [i, self.sd.stats.get_count_rank(gene)],
                [1, 0],
            )
        plt.axis('off')
        ax2.set_ylabel(' ')

        ax3 = plt.subplot(313)
        self.sd.plot_bars(ax3, color='grey')
        ax3.invert_yaxis()
        ax3.xaxis.tick_top()
        ax3.xaxis.set_label_position('top')
        ax3.set_ylabel('log(count) spatial')

    def score_affinity(self,labels_1,labels_2=None,scanpy_obs_label='celltype'):

        if labels_2 is None:
            labels_2 = (self.adata.obs[~self.adata.obs[scanpy_obs_label].isin(labels_1)])[scanpy_obs_label]
        
        mask_1 = self.adata.obs[scanpy_obs_label].isin(labels_1)
        mask_2 = self.adata.obs[scanpy_obs_label].isin(labels_2)
    
        samples_1 = self.adata[mask_1,]
        samples_2 = self.adata[mask_2,]

        counts_1 = np.array(samples_1.X.mean(0)).flatten()
        counts_2 = np.array(samples_2.X.mean(0)).flatten()

        return np.log((counts_1+0.1)/(counts_2+0.1))

        
        clrs = np.zeros((len(self.sd),))
    
        for g in self.sd.genes:
    #         if g in adata.var.index:
            clrs[self.sd.g==g]=ratios[samples_1.var.index==g]
                
    #     extreme = np.max((ratios.max(),-ratios.min()))
        
    #     fig = plt.figure(figsize=(70,70))
    # #     fig = plt.figure(figsize=(100,100))
        
    #     ax = plt.subplot(111)
    # #     ax.imshow(1-(fitc[0:-1:5,0:-1:5].sum(-1).T**0.1),cmap='Greys')
    #     sdata.background.imshow(cmap='Greys')
    #     cax = ax.scatter(sdata.x,sdata.y,c=clrs,cmap='seismic',alpha=0.6, vmin=-extreme,vmax=extreme)

    #     ax.set_title('differential mRNA <-> celltype mappings: '+filename, fontsize=90)

    #     # Add colorbar, make sure to specify tick locations to match desired ticklabels
    #     cbar = fig.colorbar(cax, ticks=[extreme, -extreme])
    #     cbar.ax.set_yticklabels(['\n'.join(labels0),'\n'.join(labels1),],fontsize=60)
        
    #     handlers = [plt.scatter([],[],color='k') for f in (meta.columns)]
    #     plt.legend(handlers,[f"{idx}:{meta[idx].values[0]}" for i,idx in enumerate(meta.columns)],
    #             prop={'size': 60},loc='upper left')                              
                                


        return None

class GeneStatistics(pd.DataFrame):
    def __init__(self, *args,**kwargs):
        super(GeneStatistics, self).__init__(*args,**kwargs)

    @property
    def counts_sorted(self):
        return self.data.counts[self.stats.count_indices]

    @property
    def genes(self):
        return self.index

    def get_count(self, gene):
        if gene in self.genes.values:
            return int(self.counts[self.genes == gene])

    def get_id(self, gene_name):
        return int(self.gene_ids[self.genes == gene_name])

    def get_count_rank(self, gene):
        if gene in self.genes.values:
            return int(self.count_ranks[self.genes == gene])

class PointStatistics(GeneStatistics):
    def __init__(self, sd):
        genes, indicers, inverse, counts = np.unique(
            sd['g'],
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )

        count_idcs = np.argsort(counts)
        count_ranks = np.argsort(count_idcs)

        super(PointStatistics, self).__init__(
            {
                'counts': counts,
                'count_ranks': count_ranks,
                'count_indices': count_idcs,
                'gene_ids': np.arange(len(genes))
            },
            index=genes)

        sd['gene_id'] = inverse

        sd.graph = SpatialGraph(self)

class ScStatistics(GeneStatistics):

    def __init__(self, scanpy_df):

        counts = np.array(scanpy_df.adata.X.sum(0)).squeeze()
        genes = scanpy_df.adata.var.index

        count_idcs = np.argsort(counts)
        count_ranks = np.argsort(count_idcs)

        super(ScStatistics, self).__init__(
            {
                'counts': counts,
                'count_ranks': count_ranks,
                'count_indices': count_idcs,
                'gene_ids': np.arange(len(genes))
            },
            index=genes)
  
class SpatialIndexer():

    def __init__(self, df):
        self.df = df

    @property
    def shape(self):
        if self.df.background is None:
            return np.ceil(self.df.x.max() - self.df.x.min()).astype(
                int), np.ceil(self.df.y.max() - self.df.y.min()).astype(int)
        else:
            return self.df.background.shape

    def create_cropping_mask(self, start, stop, series):

        if start is None:
            start = 0

        if stop is None:
            stop = series.max()

        return ((series > start) & (series < stop))

    def join_cropping_mask(self, xlims, ylims):
        return self.create_cropping_mask(
            *xlims, self.df.x) & self.create_cropping_mask(*ylims, self.df.y)

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

        return SpatialData(self.df.g[mask],
                           self.df.x[mask] - start_x,
                           self.df.y[mask] - start_y, pixel_maps,
                           self.df.scanpy.adata, self.df.synchronize)

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
                 G,
                 x_coordinates,
                 y_coordinates,
                 pixel_maps=[],
                 scanpy=None,
                 synchronize=True):

        # Initiate 'own' spot data:
        super(SpatialData, self).__init__({
            'g': G,
            'x': x_coordinates,
            'y': y_coordinates
        })

        self['g']=self['g'].astype('category')

        # Initiate pixel maps:
        self.pixel_maps = []
        self.stats = PointStatistics(self)

        self.graph = SpatialGraph(self)

        for pm in pixel_maps:
            if not type(pm) == PixelMap:
                self.pixel_maps.append(PixelMap(pm))
            else:
                self.pixel_maps.append(pm)

        self.synchronize = synchronize

        # Append scanpy data set, synchronize both:
        if scanpy is not None:
            self.scanpy = ScanpyDataFrame(self, scanpy)
            if self.synchronize:
                self.sync_scanpy()
        else:
            self.scanpy=None

        # self.obsm = {"spatial":np.array(self.coordinates).T}
        # self.obs = pd.DataFrame({'gene':self.g})
        self.uns={}

    @property
    def gene_ids(self):
        return self.gene_id

    @property
    def coordinates(self):
        return (self.x,self.y)

    @property
    def counts(self):
        return self.stats['counts']

    @property
    def counts_sorted(self):
        return self.stats.counts[self.stats.count_indices]

    @property
    def genes_sorted(self):
        return self.genes[self.stats.count_indices]

    @property
    def genes(self):
        return self.stats.index

    @property
    def spatial(self):
        return SpatialIndexer(self)

    @property
    def background(self):
        if len(self.pixel_maps):
            return self.pixel_maps[0]
    @property
    def adata(self):
        if self.scanpy is not None:
            return self.scanpy.adata

    @property
    def X(self):
        return scipy.sparse.csc_matrix((np.ones(len(self.g),),(np.arange(len(self.g)),np.array(self.gene_ids).flatten())),
                        shape=(len(self.g),self.genes.shape[0],))

    @property
    def var(self):
        return pd.DataFrame(index=self.stats.genes)

    @property
    def obs(self):
        return  pd.DataFrame({'gene':self.g}).astype(str).astype('category')

    @property
    def obsm(self):
        return {"spatial":np.array(self.coordinates).T}

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

            if self.scanpy is not None:
                scanpy = self.scanpy.adata
                synchronize = self.scanpy.synchronize
            else:
                scanpy = None
                synchronize = None

            new_frame = SpatialData(new_data.g,
                                    new_data.x,
                                    new_data.y,
                                    self.pixel_maps,
                                    scanpy=scanpy,
                                    synchronize=synchronize)
            # new_frame.update_stats()
            return (new_frame)

        print('Reverting to generic Pandas.')
        return super().__getitem__(*arg)

    def sync_scanpy(self,
                    mRNA_threshold_sc=1,
                    mRNA_threshold_spatial=1,
                    verbose=False,
                    anndata=None):
        if anndata is None and self.scanpy is None:
            print('Please provide some scanpy data...')

        if anndata is not None:
            self.scanpy = ScanpyDataFrame(anndata)
        else:
            self.scanpy.synchronize()

    def get_id(self, gene_name):
        return int(self.stats.gene_ids[self.genes == gene_name])


    def scatter(self,
                c=None,
                color=None,
                legend  =None,
                axd=None,
                plot_bg=True,
                cmap='jet',
                **kwargs):

        if axd is None:
            axd = plt.subplot(111)

        if self.background and plot_bg:
            self.background.imshow(cmap='Greys', axd=axd)

        if c is None and color is None:
            c = self.gene_ids

        cmap = get_cmap(cmap)    

        clrs = [cmap(f) for f in np.linspace(0,1,len(self.genes))]
        handles = [plt.scatter([],[],color=c) for c in clrs]
        if legend:
            plt.legend(handles,self.genes)

        # axd.set_title(gene)
        axd.scatter(self.x,
                    self.y,
                    c=c,
                    color=color,
                    cmap=cmap,
                    **kwargs)

    def plot_bars(self, axis=None, **kwargs):
        if axis is None:
            axis = plt.subplot(111)
        axis.bar(np.arange(len(self.stats.counts)), self.counts_sorted,
                 **kwargs)
        axis.set_yscale('log')

        axis.set_xticks(
            np.arange(len(self.genes_sorted)),
            self.genes_sorted,
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
            gene = self.genes[idx]
            plot_name = 'scatter_' + str(i + 1)
            axd[plot_name].set_title(gene)
            axd[plot_name].scatter(self.x, self.y, color=(0.5, 0.5, 0.5, 0.1))
            axd[plot_name].scatter(self.x[self['gene_id'] == idx],
                                   self.y[self['gene_id'] == idx],
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
        # distances, _, _ = self.knn(n_neighbors=n_neighbors)
        self.graph.update_knn(n_neighbors=n_neighbors)
        distances = self.graph.distances
        plt.hist(distances[:, 1:n_neighbors].flatten(), **kwargs)

    def spatial_decomposition(
        self,
        mRNAs_center=None,
        mRNAs_neighbor=None,
        n_neighbors=10,
    ):

        if mRNAs_center is None:
            mRNAs_center = self.genes
        if mRNAs_neighbor is None:
            mRNAs_neighbor = self.genes

        self.graph.update_knn(n_neighbors=n_neighbors)
        neighbors = self.graph.neighbors
        # np.array(self.gene_ids)[neighbors]
        neighbor_classes = self.graph.neighbor_types

        pptx = []
        ppty = []
        clrs = []
        intensity = []

        out = np.zeros((30, 30, len(self.genes)))

        mask_center = np.logical_or.reduce(
            [neighbor_classes[:, 0] == self.get_id(m) for m in mRNAs_center])

        for i_n_neighbor in range(1, n_neighbors):

            mask_neighbor = np.logical_or.reduce([
                neighbor_classes[:, i_n_neighbor] == self.get_id(m)
                for m in mRNAs_neighbor
            ])

            mask_combined = np.logical_and(mask_center, mask_neighbor)

            for i_neighbor, n in enumerate(neighbors[mask_combined]):

                xs = np.array(self.x.iloc[n])
                ys = np.array(self.y.iloc[n])

                x_centered = xs - xs[0]
                y_centered = ys - ys[0]

                loc_neighbor = np.array(
                    (x_centered[i_n_neighbor], y_centered[i_n_neighbor]))
                loc_neighbor_normalized = loc_neighbor / (loc_neighbor **
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

                # mask = (pptx[-1] >= -50) & (pptx[-1] < 50) & (
                #     ppty[-1] >= -50) & (ppty[-1] < 50)
                # out[ppty[-1].astype(int)[mask], pptx[-1].astype(int)[mask],
                #     clrs[-1][mask]] += 1  # intensity[-1][mask]

        #         break

        pptx = np.concatenate(pptx)
        ppty = np.concatenate(ppty)

        pptt = np.arctan(pptx / ppty)
        pptr = (pptx**2 + ppty**2)**0.5

        clrs = np.concatenate(clrs)
        
        scale = pptr.max()
        for i in range(len(mRNAs_neighbor)):
            mask = clrs==i
            out[(pptt[mask]/1.5*100).astype(int),(pptr[mask]/scale*100).astype(int),i]+=1

        plt.axhline(0)
        plt.axvline(0)
        plt.scatter(pptt, pptr, c=clrs, cmap='nipy_spectral', alpha=0.1, s=3)

        # out[(pptr/pptr.max()*29).astype(int),(pptt/np.pi*2*29).astype(int),clrs]+=1
        # for u in np.unique(clrs):
        #     plt.scatter(pptx[clrs == u], ppty[clrs == u], alpha=0.1)
        # plt.scatter(
        #     0,
        #     0,
        #     color='lime',
        #     s=70,
        # )

        # intensity = np.concatenate(intensity)
        return (pptt, pptr, clrs,)

    def knn_clean(
        self,
        n_neighbors=10,
    ):
        # distances, indices, types = self.knn(n_neighbors=n_neighbors)
        self.graph.update_knn(n_neighbors=n_neighbors)
        types = self.graph.neighbor_types
        count_matrix = sparse.lil_matrix(
            (types.shape[0], self.genes.shape[0]))
        for i, t in enumerate(types):
            classes, counts = (np.unique(t[:n_neighbors], return_counts=True))
            count_matrix[i, classes] = counts / counts.sum()

        count_matrix = count_matrix.tocsr()

        count_matrix_log = count_matrix.copy()
        count_matrix_log.data = np.log(count_matrix.data)
        count_matrix_inv = count_matrix.copy()
        count_matrix_inv.data = 1 / (count_matrix.data)

        prototypes = np.zeros((len(self.genes), ) * 2)
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

    def scatter_celltype_affinities(self,
                                    adata,
                                    celltypes_1,
                                    celltypes_2=None):
        adata, sdata = synchronize(adata, self)

    def squidpy(self):
        # obs={"cluster":self.gene_id.astype('category')}
        obsm = {"spatial":np.array(self.coordinates).T}
        # var= self.genes
        # self.obs = self.index
        # X = self.X #scipy.sparse.csc_matrix((np.ones(len(self.g),),(np.arange(len(self.g)),np.array(self.gene_ids).flatten())),
                        # shape=(len(self.g),self.genes.shape[0],))

        # sparse_representation = scipy.sparse.scr()
        # var = self.var #pd.DataFrame(index=self.genes)
        uns = self.uns.update({'Image':self.background})
        obs = pd.DataFrame({'gene':self.g})
        obs['gene']=obs['gene'].astype('category')
        return  anndata.AnnData(X=self.X,obs=obs,var=self.var,obsm=obsm)

## here starts plotting.py

def create_colorarray(sdata,values,cmap=None):
    if cmap is None:
        return values[sdata.gene_ids]



def determine_text_coords(embedding, sdata,untangle_rounds=10, min_distance=0.5):
    knn = NearestNeighbors(n_neighbors=50)
    knn.fit(embedding)
    distances,neighbors = knn.kneighbors(embedding)
    neighbor_types = np.array(sdata.gene_ids)[neighbors]
    cogs = []

    for i in range(len(sdata.stats)):
        nt_filtered = 1/(1+distances)
        nt_filtered[(neighbor_types[:,0]!=i)]=0
        nt_filtered[(neighbor_types!=i)]=0
        gravity = nt_filtered
        
        gravity = gravity.sum(1)
        cog = gravity.argmax()
        cogs.append(embedding[cog])

    cogs = np.array(cogs)

    knn = NearestNeighbors(n_neighbors=2)
    
    cogs_new = cogs.copy()

    for i in range(untangle_rounds):

        cogs = cogs_new.copy()
        knn = NearestNeighbors(n_neighbors=2)

        knn.fit(cogs)
        distances,neighbors = knn.kneighbors(cogs)
        too_close = (distances[:,1]<min_distance)

        for i,c in enumerate(np.where(too_close)[0]):
            partner = neighbors[c,1]
            cog = cogs[c]-cogs[partner]
            cog_new = cogs[c]+0.3*cog
            cogs_new[c]= cog_new
            
    return cogs_new


def hbar_compare(stat1,stat2,labels=None,text_display_threshold=0.02):

    genes_united=sorted(list(set(np.concatenate([stat1.index,stat2.index]))))[::-1]
    counts_1=[0]+[stat1.loc[i].counts if i in stat1.index else 0 for i in genes_united]
    counts_2=[0]+[stat2.loc[i].counts if i in stat2.index else 0 for i in genes_united]
    cum1 = np.cumsum(counts_1)/sum(counts_1)
    cum2 = np.cumsum(counts_2)/sum(counts_2)

    for i in range(1,len(cum1)):

        bars = plt.bar([0,1],[cum1[i]-cum1[i-1],cum2[i]-cum2[i-1]],
                bottom=[cum1[i-1],cum2[i-1],], width=0.4)
        clr=bars.get_children()[0].get_facecolor()
        plt.plot((0.2,0.8),(cum1[i],cum2[i]),c='grey')
        plt.fill_between((0.2,0.8),(cum1[i],cum2[i]),(cum1[i-1],cum2[i-1]),color=clr,alpha=0.2)
        
        if (counts_1[i]/sum(counts_1)>text_display_threshold) or \
        (counts_2[i]/sum(counts_2)>text_display_threshold): 
            plt.text(0.5,(cum1[i]+cum1[i-1]+cum2[i]+cum2[i-1])/4,
                    genes_united[i-1],ha='center',)  

    if labels is not None:
        plt.xticks((0,1),labels) 

        
# def determine_gains(sc, spatial):

#     sc_genes = sc.var.index
#     counts_sc = np.array(sc.x.sum(0) / sc.x.sum()).flatten()
#     counts_spatial = np.array([spatial.get_count(g) for g in sc_genes])
#     counts_spatial = counts_spatial / counts_spatial.sum()
#     count_ratios = counts_sc / counts_spatial
#     return count_ratios


# def plot_gains(sc, spatial):

#     sc_genes = sc.var.index

#     # counts_spatial_reindexed = counts_spatial [[spatial.get_sort_index(g) for g in sc.unique_genes_sorted]]
#     count_ratios = determine_gains(sc, spatial)
#     idcs = np.argsort(count_ratios)

#     ax = plt.subplot(111)

#     span = np.linspace(0, 1, len(idcs))
#     clrs = np.stack([
#         span,
#         span * 0,
#         span[::-1],
#     ]).T

#     ax.barh(range(len(count_ratios)), np.log(count_ratios[idcs]), color=clrs)

#     ax.text(0,
#             len(idcs) + 3,
#             'lost in spatial ->',
#             ha='center',
#             fontsize=12,
#             color='red')
#     ax.text(0, -3, '<- lost in SC', ha='center', fontsize=12, color='lime')

#     for i, gene in enumerate(sc_genes[idcs]):
#         if count_ratios[idcs[i]] < 1:
#             ha = 'left'
#             xpos = 0.05
#         else:
#             ha = 'right'
#             xpos = -0.05
#         ax.text(0, i, gene, ha=ha)

#     ax.set_yticks([], [])


# def compare_counts(sc, spatial):

#     sc_genes = (sc.var.index)
#     sc_counts = (np.array(sc.x.sum(0)).flatten())
#     sc_count_idcs = np.argsort(sc_counts)
#     count_ratios = np.log(determine_gains(sc, spatial))
#     count_ratios -= count_ratios.min()
#     count_ratios /= count_ratios.max()

#     ax1 = plt.subplot(311)
#     ax1.set_title('compared molecule counts:')
#     ax1.bar(np.arange(len(sc_counts)), sc_counts[sc_count_idcs], color='grey')
#     ax1.set_ylabel('log(count) scRNAseq')
#     ax1.set_xticks(np.arange(len(sc_genes)),
#                    sc_genes[sc_count_idcs],
#                    rotation=90)
#     ax1.set_yscale('log')

#     ax2 = plt.subplot(312)
#     for i, gene in enumerate(sc_genes[sc_count_idcs]):
#         plt.plot(
#             [i, spatial.get_count_rank(gene)],
#             [1, 0],
#         )
#     plt.axis('off')
#     ax2.set_ylabel(' ')

#     ax3 = plt.subplot(313)
#     spatial.plot_bars(ax3, color='grey')
#     ax3.invert_yaxis()
#     ax3.xaxis.tick_top()
#     ax3.xaxis.set_label_position('top')
#     ax3.set_ylabel('log(count) spatial')


# def get_count(adata, gene):
#     return (adata.x[:, adata.var.index == gene].sum())


# def compare_counts_cellwise(sc, spatial, cell_obs_label):

#     genes = list(sc.unique_genes)
#     count_ratios = np.log(determine_gains(sc, spatial))
#     count_ratios -= count_ratios.min()
#     count_ratios /= count_ratios.max()

#     ax1 = plt.subplot(311)

#     # print([genes.index(g) for g in sc.unique_genes])

#     span = np.array(
#         [count_ratios[sc.get_index(g)] for g in sc.unique_genes_sorted])
#     clrs = np.stack([
#         span,
#         span * 0,
#         1 - span,
#     ]).T

#     ax1.set_title('compared molecule counts:')
#     sc.plot_bars(ax1, color=clrs)
#     ax1.set_ylabel('log(count) scRNAseq')

#     ax2 = plt.subplot(312)
#     for i, gene in enumerate(sc.unique_genes_sorted):
#         plt.plot(
#             [i, spatial.get_sort_index(gene)],
#             [1, 0],
#         )
#     plt.axis('off')
#     ax2.set_ylabel(' ')

#     ax3 = plt.subplot(313)

#     span = np.array(
#         [count_ratios[sc.get_index(g)] for g in spatial.unique_genes_sorted])

#     clrs = np.stack([
#         span,
#         span * 0,
#         1 - span,
#     ]).T

#     spatial.plot_bars(ax3, color=clrs)
#     ax3.invert_yaxis()
#     ax3.xaxis.tick_top()
#     ax3.xaxis.set_label_position('top')
#     ax3.set_ylabel('log(count) spatial')


# def plot_overview(spatial):
#     spatial.plot_overview()


# def synchronize(sc,
#                 spatial,
#                 mRNA_threshold_sc=10,
#                 mRNA_threshold_spatial=10,
#                 verbose=True):

#     genes_sc = set(sc.var.index)
#     genes_spatial = set(spatial.genes)

#     genes_xor = genes_sc.symmetric_difference(genes_spatial)

#     for gene in genes_xor:
#         if gene in genes_sc:
#             pass
#             # print('Removing {}, which is not present in spatial data'.format(
#             #     gene))
#         else:
#             print('Removing {}, which is not present in SC data'.format(gene))

#     genes_and = (genes_sc & genes_spatial)

#     for gene in sorted(genes_and):
#         if verbose:
#             print(gene)
#         if (get_count(sc, gene) < mRNA_threshold_sc):
#             if verbose:
#                 print('Removing {}, low count in SC data'.format(gene))
#             genes_and.remove(gene)
#         elif spatial.get_count(gene) < mRNA_threshold_spatial:
#             if verbose:
#                 print('Removing {}, low count in spatial data'.format(gene))
#             genes_and.remove(gene)

#     genes_and = sorted(genes_and)

#     # spatial.filter(genes_and)
#     sc = sc[:, genes_and]

#     return (sc, spatial[spatial.g.isin(genes_and)])
