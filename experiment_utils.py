import numpy as np
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

def plot_evaluation(ax, descriptor_list, data_dict, x_values, x_label, y_label, title):
    linestyles = [('.', 'solid'), ('o', 'dotted'), ('^', 'dashed'), ('s', 'dashdot'), ('o', 'solid'), ('.', 'dotted'),
                  ('s', 'dashed'), ('^', 'dashdot')]

    # Plotting
    for index, descriptor in enumerate(descriptor_list):
        if len(descriptor) > 1:
            legend_label = "Desc. Combination"
        else:
            legend_label = descriptor[0].replace("_", " ").capitalize()

        linestyle = linestyles[index % len(linestyles)]
        ax.plot(x_values, data_dict["_".join(descriptor)],
                label=legend_label, linestyle=linestyle[1],
                marker=linestyle[0])

    ax.set_xlabel(x_label, color='white')
    ax.set_ylabel(y_label, color='white')
    ax.set_title(title, color='white', loc="left")
    ax.legend(loc="upper left")
    ax.grid(True)

    ax.set_facecolor('#333333')  # Dark grey background for the plot area

    # Customize ticks and labels
    ax.tick_params(colors='white')  # White ticks
    ax.xaxis.label.set_color('white')  # X-axis label color
    ax.yaxis.label.set_color('white')  # Y-axis label color

    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    ax.set_xlim(np.min(x_values), np.max(x_values))

    # Customize grid
    ax.grid(color='#666666', linestyle='--', linewidth=0.7)
