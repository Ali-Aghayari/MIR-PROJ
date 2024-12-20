import string

import fasttext
import numpy as np
import os

from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
from Logic.core.word_embedding.fasttext_model import FastText
from Logic.core.clustering.dimension_reduction import DimensionReduction
from Logic.core.clustering.clustering_metrics import ClusteringMetrics
from Logic.core.clustering.clustering_utils import ClusteringUtils


# Main Function: Clustering Tasks

def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                    punctuation_removal=True):
    if lower_case: text = text.lower()
    tokenized = [i for i in word_tokenize(text) if len(i) >= minimum_length]
    if punctuation_removal: tokenized = [i for i in tokenized if i not in string.punctuation]
    if stopword_removal: tokenized = [i for i in tokenized if
                                      i not in set(stopwords.words('english')).union(set(stopwords_domain))]
    text = " ".join(tokenized)
    return text


if __name__ == "__main__":
    # TODO ::::::: note : couldn't connect myself to wandb site for debugging and seeing the plots so i just used plot show!

    # 0. Embedding Extraction
    # TODO: Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.

    model = fasttext.load_model("C:/Users/Ali/PycharmProjects/MIR-PROJ/Logic/core/word_embedding/FastText_model.bin")

    path = './Phase_1/index/'
    ft_data_loader = FastTextDataLoader(path)
    X, y = ft_data_loader.create_train_data()

    X = X[:4000]
    y = y[:4000]

    embeddings = []
    for i in X:
        embeddings.append(model.get_sentence_vector(i))
    embeddings = np.array(embeddings)
    y = np.array(y)

    # 1. Dimension Reduction
    # TODO: Perform Principal Component Analysis (PCA):
    #     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
    #     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
    #     - Draw plots to visualize the results.
    DR = DimensionReduction()

    DR.wandb_plot_explained_variance_by_components(embeddings, "clusters", "variance by components")
    # TODO : note : from plot : about 80 percent is preserved
    n_components = 2
    reduced, SV, EVR = DR.pca_reduce_dimension(embeddings, n_components)
    """
    
    plt.figure(figsize=(10, 7))
    plt.bar(range(1, n_components + 1), EVR, alpha=0.7, align='center')
    plt.step(range(1, n_components + 1), np.cumsum(EVR), where='mid')
    plt.ylabel('EVR')
    plt.xlabel('p-components')
    plt.show()
    
    ... 
    """

    # TODO: Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
    #     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
    #     - Use the output vectors from this step to draw the diagram.
    DR.wandb_plot_2d_tsne(reduced, "clusters", "tsne")

    # 2. Clustering
    ## K-Means Clustering
    # TODO: Implement the K-means clustering algorithm from scratch.
    # TODO: Create document clusters using K-Means.
    # TODO: Run the algorithm with several different values of k.
    # TODO: For each run:
    #     - Determine the genre of each cluster based on the number of documents in each cluster.
    #     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
    #     - Check the implementation and efficiency of the algorithm in clustering similar documents.
    # TODO: Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
    # TODO: Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)

    #reduced = DR.pca_reduce_dimension(embeddings, 4)
    CU = ClusteringUtils()

    all_k = list(range(2, 10))
    for i in all_k:
        CU.visualize_kmeans_clustering_wandb(reduced, i, "clusters", "kmeans")

    CU.plot_kmeans_cluster_scores(reduced, y, all_k, "clusters", "kmeans scores")
    CU.visualize_elbow_method_wcss(reduced, all_k, "clusters", "elbow")



    ## Hierarchical Clustering
    # TODO: Perform hierarchical clustering with all different linkage methods.
    # TODO: Visualize the results.
    CU.wandb_plot_hierarchical_clustering_dendrogram(reduced, "clusters", "ward", "ward dendrogram")
    CU.wandb_plot_hierarchical_clustering_dendrogram(reduced, "clusters", "complete", "complete dendrogram")
    CU.wandb_plot_hierarchical_clustering_dendrogram(reduced, "clusters", "average", "average dendrogram")
    CU.wandb_plot_hierarchical_clustering_dendrogram(reduced, "clusters", "single", "single dendrogram")

    # 3. Evaluation
    # TODO: Using clustering metrics, evaluate how well your clustering method is performing.
    CM = ClusteringMetrics()

    print("scores for kmeans:")
    centers, labels = CU.cluster_kmeans(reduced, 4)
    S, P, A = CM.silhouette_score(reduced, labels), CM.purity_score(y, labels), CM.adjusted_rand_score(y, labels)
    print(f'silhouette : {S} Purity : {P} Adjusted Rand : {A}')

    print("scores for H-ward:")
    labels = CU.cluster_hierarchical_ward(reduced)
    S, P, A = CM.silhouette_score(reduced, labels), CM.purity_score(y, labels), CM.adjusted_rand_score(y, labels)
    print(f'silhouette : {S} Purity : {P} Adjusted Rand : {A}')

    print("scores for H-complete:")
    labels = CU.cluster_hierarchical_complete(reduced)
    S, P, A = CM.silhouette_score(reduced, labels), CM.purity_score(y, labels), CM.adjusted_rand_score(y, labels)
    print(f'silhouette : {S} Purity : {P} Adjusted Rand : {A}')

    print("scores for H-average:")
    labels = CU.cluster_hierarchical_average(reduced)
    S, P, A = CM.silhouette_score(reduced, labels), CM.purity_score(y, labels), CM.adjusted_rand_score(y, labels)
    print(f'silhouette : {S} Purity : {P} Adjusted Rand : {A}')

    print("scores for H-single:")
    labels = CU.cluster_hierarchical_average(reduced)
    S, P, A = CM.silhouette_score(reduced, labels), CM.purity_score(y, labels), CM.adjusted_rand_score(y, labels)
    print(f'silhouette : {S} Purity : {P} Adjusted Rand : {A}')
