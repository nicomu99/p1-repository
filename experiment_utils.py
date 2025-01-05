import os
import ast
import pickle
import numpy as np
import pandas as pd
from descriptor_utils import DescriptorWrapper
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


def compute_knn(data_df, descriptor_list=None, variant='directed', mode='connectivity', n_neighbors=5, pca=False,
                pca_dim=3):
    """
    Works as a wrapper for creating k-NN graphs based on the requirement that graph properties can be changeable.
    The function can return directed, symmetric and mutual graphs, as well as directed and undirected ones. The default number
    of neighbors was chosen arbitrarily.

    :param data_df: Pandas Dataframe
        A dataframe containing the objects the k-NN graphs should be calculated on.
    :param descriptor_list: List of Strings
        A list used to specify which descriptors should be used to calculate the k-NN graph. The list should contain
        'sirm', 'scomp', 'evrap', 'samp', 'pfh', 'sector_model', 'shell_model', or 'combined_model'.
    :param variant: {'directed', 'symmetric', 'mutual'}, default: 'directed'
        Used to specify what type of k-NN graph should be created. Mutual k-NN graphs only contain an edge, if both A is a neighbor of B and B is a neighbor of A, the lower distance is inserted into the graph in this implementation. A symmetric k-NN graph represents a logical or: If A is a neighbor of B or B is a neighbor of A, then B is a neighbor of A and A is a neighbor of B. In this implementation, for a weighted graph, the larger distance is used. A directed graph means that A can be a neighbor of B, even if B is not a neighbor of A.
    :param mode: {'connectivity', 'distance'}, default: 'connectivity'
        Choose if the k-NN graph is weighted or unweighted. 'connectivity' means the graph is unweighted, whereas 'distance' means the graph is weighted.
    :param n_neighbors: int, default: 5
        Specify how many nearest neighbors per object should be in the k-NN graph.
    :param pca: bool
        True if PCA should be performed on the data before calculating the knn
    :param pca_dim: int
        The dimensionality of PCA
    :return: ndarray
        A k-NN graph.
    """
    if descriptor_list is None:
        descriptor_list = ['evrap']

    knn_data = []
    if 'evrap' in descriptor_list:
        knn_data.append(np.stack(data_df['evrap'].to_numpy()))
    if 'samp' in descriptor_list:
        knn_data.append(np.stack(data_df['samp'].to_numpy()))
    if 'sirm' in descriptor_list:
        knn_data.append(np.stack(data_df['sirm'].to_numpy()).reshape(len(data_df), 1))
    if 'scomp' in descriptor_list:
        knn_data.append(np.stack(data_df['scomp'].to_numpy()).reshape(len(data_df), 1))
    if 'samp_3d' in descriptor_list:
        knn_data.append(np.stack(data_df['samp_3d'].to_numpy()))
    if 'sirm_3d' in descriptor_list:
        knn_data.append(np.stack(data_df['sirm_3d'].to_numpy()))
    if 'scomp_3d' in descriptor_list:
        knn_data.append(np.stack(data_df['scomp_3d'].to_numpy()))

    if 'pfh' in descriptor_list:
        knn_data.append(np.stack(data_df['pfh'].to_numpy()))
    if 'sector_model' in descriptor_list:
        knn_data.append(np.stack(data_df['sector_model'].to_numpy()))
    if 'shell_model' in descriptor_list:
        knn_data.append(np.stack(data_df['shell_model'].to_numpy()))
    if 'combined_model' in descriptor_list:
        knn_data.append(np.stack(data_df['combined_model'].to_numpy()))

    knn_data = np.hstack(knn_data)

    if pca:
        pca_dim = np.min([knn_data.shape[1], pca_dim])
        pca = PCA(n_components=pca_dim)
        knn_data = pca.fit_transform(knn_data)

    # Graph is a matrix with entry A[i, j] if j is in the k-NN graph of i
    graph = kneighbors_graph(knn_data, mode=mode, n_neighbors=n_neighbors, n_jobs=-1)
    graph = graph.toarray()

    # the standard output of kneighbors_graph is a directed k-NN graph
    if variant == 'symmetric':
        graph = np.maximum(graph, graph.T).astype(float)
    elif variant == 'mutual':
        graph = np.minimum(graph, graph.T).astype(float)

    return graph

def plot_evaluation(ax, descriptor_list, data_dict, x_values, x_label, y_label, title, colors=None, log_y=False):
    linestyles = [('.', 'solid'), ('o', 'dotted'), ('^', 'dashed'), ('s', 'dashdot'), ('o', 'solid'), ('.', 'dotted'),
                  ('s', 'dashed'), ('^', 'dashdot')]

    # Plotting
    for index, descriptor in enumerate(descriptor_list):
        if len(descriptor) > 1:
            legend_label = "Desc. Combination"
        else:
            legend_label = descriptor[0].replace("_", " ").capitalize()

        linestyle = linestyles[index % len(linestyles)]
        plot_args = {
            'label': legend_label,
            'linestyle': linestyle[1],
            'marker': linestyle[0]
        }
        if colors is not None:
            plot_args['color'] = colors[index]

        ax.plot(x_values, data_dict["_".join(descriptor)], **plot_args)

    ax.set_xlabel(x_label, color='white')
    ax.set_ylabel(y_label, color='white')
    ax.set_title(title, color='white', loc="left")
    ax.legend(loc="upper left")
    ax.grid(True)

    if log_y:
        ax.set_yscale('log')
    else:
        max_y = max(i for v in data_dict.values() for i in v)
        ax.set_ylim(bottom=0, top=max_y * 1.2)

    ax.set_facecolor('#333333')

    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    ax.set_xlim(np.min(x_values), np.max(x_values))

    ax.grid(color='#666666', linestyle='--', linewidth=0.7)

def create_rotation_matrix(rng):
    theta_x = rng.uniform(0, 2 * np.pi)
    theta_y = rng.uniform(0, 2 * np.pi)
    theta_z = rng.uniform(0, 2 * np.pi)

    r_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

    r_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    r_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    return r_z @ r_y @ r_x

def randomly_rotate_point_clouds(point_clouds):
    rng = np.random.default_rng(seed=42)

    if type(point_clouds) == np.ndarray:
        rotated_pcs = np.empty_like(point_clouds)

        for i in range(point_clouds.shape[0]):
            rotation_matrix = create_rotation_matrix(rng)
            rotated_pcs[i] = np.dot(point_clouds[i], rotation_matrix.T)
    else:
        rotated_pcs = []
        for pc in point_clouds:
            rotation_matrix = create_rotation_matrix(rng)
            rotated_pcs.append(np.dot(pc, rotation_matrix.T))

    return rotated_pcs

def compute_descriptors_from_file(file_name, rotate_random=False):
    descriptor_list = [
        'evrap', 'sirm', 'scomp', 'samp', 'sector_model', 'shell_model', 'combined_model',
        'pfh', 'sirm_3d', 'scomp_3d', 'samp_3d'
    ]

    file = f"test_output/{file_name}{'_rotated' if rotate_random else ''}.csv"
    if os.path.isfile(file):
        df = pd.read_csv(file, index_col=0)

        for col in descriptor_list:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x)  else x)

        return df
    else:
        if not file == 'mc_gill_whole':
            with open(f"point_clouds/{file_name}.pkl", "rb") as f:
                data = pickle.load(f)
        else:
            data = np.load(f"point_clouds/{file_name}.npz", allow_pickle=True)
        point_clouds = data['objects']
        labels = data['labels']

        if rotate_random:
            point_clouds = randomly_rotate_point_clouds(point_clouds)

        descriptor_wrapper = DescriptorWrapper()
        descriptor_embeddings = dict()

        for descriptor in descriptor_list:
            desc_output = descriptor_wrapper.compute_model_on_dataset(point_clouds, descriptor)
            # need to convert for saving and loading
            desc_output = desc_output.tolist()
            descriptor_embeddings[descriptor] = desc_output

        df = pd.DataFrame({k: list(v) for k, v in descriptor_embeddings.items()})
        df["labels"] = labels
        df.to_csv(file)

        return df




def compute_ratio_cut(adj_list, clusters):
    unique_labels = np.unique(clusters)

    # Compute ratio cut value
    ratio_cut_value = 0
    for cluster in unique_labels:
        # Find nodes in the current cluster
        cluster_nodes = np.where(clusters == cluster)[0]
        cluster_size = len(cluster_nodes)

        # Find nodes not in the current cluster
        complement_nodes = np.setdiff1d(np.arange(len(clusters)), cluster_nodes)

        cut_value = adj_list[np.ix_(cluster_nodes, complement_nodes)].sum()

        # Add normalized cut to the total
        ratio_cut_value += cut_value / cluster_size

    return ratio_cut_value


def compute_normalized_cut(adj_matrix, clusters):
    unique_labels = np.unique(clusters)

    normalized_cut_value = 0
    for cluster in unique_labels:
        # Find nodes in the current cluster
        cluster_nodes = np.where(clusters == cluster)[0]

        # Find nodes not in the current cluster
        complement_nodes = np.setdiff1d(np.arange(len(clusters)), cluster_nodes)

        # Compute Cut(V_i, complement)
        cut_value = adj_matrix[np.ix_(cluster_nodes, complement_nodes)].sum()

        # Compute Vol(V_i)
        vol_value = adj_matrix[cluster_nodes, :].sum()

        # Add normalized cut value
        if vol_value > 0:  # Avoid division by zero
            normalized_cut_value += cut_value / vol_value

    return normalized_cut_value
