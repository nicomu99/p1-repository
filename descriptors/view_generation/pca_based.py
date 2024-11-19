from sklearn.decomposition import PCA


def pca_attributes(vertices_list):
    pca_singular_values = []
    pca_explained_variance = []
    pca_explained_variance_ratio = []

    for vertices in vertices_list:
        pca = PCA(n_components=3)
        pca.fit(vertices)
        pca_explained_variance.append(pca.explained_variance_)
        pca_explained_variance_ratio.append(pca.explained_variance_ratio_)
        pca_singular_values.append(pca.singular_values_)

    return pca_singular_values, pca_explained_variance, pca_explained_variance_ratio


def evarp(vertices):
    pca = PCA(n_components=3)
    pca.fit(vertices)
    return pca.explained_variance_