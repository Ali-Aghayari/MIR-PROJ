import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import wandb

from typing import List, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from collections import Counter
from Logic.core.clustering.clustering_metrics import *

from Logic.core.clustering.dimension_reduction import *


class ClusteringUtils:

    def cluster_kmeans(self, emb_vecs: List, n_clusters: int, max_iter: int = 100) -> Tuple[List, List]:
        """
        Clusters input vectors using the K-means method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List]
            Two lists:
            1. A list containing the cluster centers.
            2. A list containing the cluster index for each input vector.
        """

        """KM = KMeans(n_clusters=n_clusters, random_state=0, max_iter=max_iter)
        KM.fit(emb_vecs)
        return (KM.cluster_centers_, KM.labels_)"""

        centers = [emb_vecs[i] for i in np.random.choice(len(emb_vecs), size=n_clusters, replace=False)]
        # total_distance = float('inf')
        for _ in range(max_iter):
            cluster_indices = [np.argmin([np.linalg.norm(vec - center) for center in centers]) for vec in emb_vecs]
            new_centers = [
                np.mean([emb_vecs[j] for j in range(len(emb_vecs)) if cluster_indices[j] == i], axis=0) if any(
                    cluster_indices[j] == i for j in range(len(emb_vecs))) else centers[i] for i in range(n_clusters)]
            # new_total_distance = sum(np.linalg.norm(emb_vecs[i] - new_centers[cluster_indices[i]]) ** 2 for i in range(len(emb_vecs)))
            # if np.isclose(total_distance, new_total_distance): break
            # total_distance = new_total_distance
            centers = new_centers
        return centers, cluster_indices

    def get_most_frequent_words(self, documents: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Finds the most frequent words in a list of documents.

        Parameters
        -----------
        documents: List[str]
            A list of documents, where each document is a string representing a list of words.
        top_n: int, optional
            The number of most frequent words to return. Default is 10.

        Returns
        --------
        List[Tuple[str, int]]
            A list of tuples, where each tuple contains a word and its frequency, sorted in descending order of frequency.
        """
        counter = Counter()
        for i in documents:
            counter.update(i.split())
        return counter.most_common(top_n)

    def cluster_kmeans_WCSS(self, emb_vecs: List, n_clusters: int) -> Tuple[List, List, float]:
        """ This function performs K-means clustering on a list of input vectors and calculates the Within-Cluster Sum of Squares (WCSS) for the resulting clusters.

        This function implements the K-means algorithm and returns the cluster centroids, cluster assignments for each input vector, and the WCSS value.

        The WCSS is a measure of the compactness of the clustering, and it is calculated as the sum of squared distances between each data point and its assigned cluster centroid. A lower WCSS value indicates that the data points are closer to their respective cluster centroids, suggesting a more compact and well-defined clustering.

        The K-means algorithm works by iteratively updating the cluster centroids and reassigning data points to the closest centroid until convergence or a maximum number of iterations is reached. This function uses a random initialization of the centroids and runs the algorithm for a maximum of 100 iterations.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List, float]
            Three elements:
            1) A list containing the cluster centers.
            2) A list containing the cluster index for each input vector.
            3) The Within-Cluster Sum of Squares (WCSS) value for the clustering.
        """

        """KM = KMeans(n_clusters=n_clusters, random_state=0)
        KM.fit(emb_vecs)
        return (KM.cluster_centers_, KM.labels_, KM.inertia_)"""

        centers = [emb_vecs[i] for i in np.random.choice(len(emb_vecs), size=n_clusters, replace=False)]
        total_distance, max_iter = float('inf'), 100
        for _ in range(max_iter):
            cluster_indices = [np.argmin([np.linalg.norm(vec - center) for center in centers]) for vec in emb_vecs]
            new_centers = [
                np.mean([emb_vecs[j] for j in range(len(emb_vecs)) if cluster_indices[j] == i], axis=0) if any(
                    cluster_indices[j] == i for j in range(len(emb_vecs))) else centers[i] for i in range(n_clusters)]
            new_total_distance = sum(
                np.linalg.norm(emb_vecs[i] - new_centers[cluster_indices[i]]) ** 2 for i in range(len(emb_vecs)))
            if np.isclose(total_distance, new_total_distance): break
            total_distance = new_total_distance
            centers = new_centers
        return centers, cluster_indices, total_distance

    def cluster_hierarchical_single(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with single linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        # TODO : note : hard coded 4 after watching the plots
        KM = AgglomerativeClustering(linkage='single', n_clusters=4)
        KM.fit_predict(emb_vecs)
        return KM.labels_

    def cluster_hierarchical_complete(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with complete linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        # TODO : note : hard coded 4 after watching the plots
        KM = AgglomerativeClustering(linkage='complete', n_clusters=4)
        KM.fit(emb_vecs)
        return KM.labels_

    def cluster_hierarchical_average(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with average linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        # TODO : note : hard coded 4 after watching the plots
        KM = AgglomerativeClustering(linkage='average', n_clusters=4)
        KM.fit(emb_vecs)
        return KM.labels_

    def cluster_hierarchical_ward(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with Ward's method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        # TODO : note : hard coded 4 after watching the plots
        KM = AgglomerativeClustering(linkage='ward', n_clusters=4)
        KM.fit(emb_vecs)
        return KM.labels_

    def visualize_kmeans_clustering_wandb(self, data, n_clusters, project_name, run_name):
        """ This function performs K-means clustering on the input data and visualizes the resulting clusters by logging a scatter plot to Weights & Biases (wandb).

        This function applies the K-means algorithm to the input data and generates a scatter plot where each data point is colored according to its assigned cluster.
        For visualization use convert_to_2d_tsne to make your scatter plot 2d and visualizable.
        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform K-means clustering on the input data with the specified number of clusters.
        3. Obtain the cluster labels for each data point from the K-means model.
        4. Create a scatter plot of the data, coloring each point according to its cluster label.
        5. Log the scatter plot as an image to the wandb run, allowing visualization of the clustering results.
        6. Close the plot display window to conserve system resources (optional).

        Parameters
        -----------
        data: np.ndarray
            The input data to perform K-means clustering on.
        n_clusters: int
            The number of clusters to form during the K-means clustering process.
        project_name: str
            The name of the wandb project to log the clustering visualization.
        run_name: str
            The name of the wandb run to log the clustering visualization.

        Returns
        --------
        None
        """
        # Initialize wandb
        # wandb.login(key="63279235fbf3eb23c36ab8ad68bb00ffae0a06f9")
        # run = wandb.init(project=project_name, name=run_name)

        # Perform K-means clustering
        # TODO
        centers, labels = self.cluster_kmeans(data, n_clusters)
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(data)

        # Plot the clusters
        # TODO
        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis')
        plt.title('K-means with t-SNE')
        plt.colorbar(label='Cluster Label')
        plt.show()

        # Log the plot to wandb
        # TODO
        # wandb.log({"KM clustering": plt})
        # Close the plot display window if needed (optional)
        # TODO
        # plt.close()

    def wandb_plot_hierarchical_clustering_dendrogram(self, data, project_name, linkage_method, run_name):
        """ This function performs hierarchical clustering on the provided data and generates a dendrogram plot, which is then logged to Weights & Biases (wandb).

        The dendrogram is a tree-like diagram that visualizes the hierarchical clustering process. It shows how the data points (or clusters) are progressively merged into larger clusters based on their similarity or distance.

        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform hierarchical clustering on the input data using the specified linkage method.
        3. Create a linkage matrix, which represents the merging of clusters at each step of the hierarchical clustering process.
        4. Generate a dendrogram plot using the linkage matrix.
        5. Log the dendrogram plot as an image to the wandb run.
        6. Close the plot display window to conserve system resources.

        Parameters
        -----------
        data: np.ndarray
            The input data to perform hierarchical clustering on.
        linkage_method: str
            The linkage method for hierarchical clustering. It can be one of the following: "average", "ward", "complete", or "single".
        project_name: str
            The name of the wandb project to log the dendrogram plot.
        run_name: str
            The name of the wandb run to log the dendrogram plot.

        Returns
        --------
        None
        """
        # wandb.login(key="63279235fbf3eb23c36ab8ad68bb00ffae0a06f9")
        # run = wandb.init(project=project_name, name=run_name)
        # Perform hierarchical clustering
        # TODO
        labels = None
        if linkage_method == "ward":
            labels = self.cluster_hierarchical_ward(data)
        elif linkage_method == "average":
            labels = self.cluster_hierarchical_average(data)
        elif linkage_method == "complete":
            labels = self.cluster_hierarchical_complete(data)
        elif linkage_method == "single":
            labels = self.cluster_hierarchical_single(data)
        else:
            raise Exception("Invalid linkage method")

        # Create linkage matrix for dendrogram
        # TODO
        linkage_ = linkage(data, method=linkage_method)

        # Generate a dendrogram plot using the linkage matrix.
        # TODO
        plt.figure(figsize=(10, 7))
        dendrogram_ = dendrogram(linkage_)
        plt.title(f'HC Dendrogram ({linkage_method} linkage)')
        plt.xlabel('index')
        plt.ylabel('distance')
        plt.show()

        # Log the dendrogram plot as an image to the wandb run.
        # TODO
        # img_path = 'HC dendrogram.png'
        # plt.savefig(img_path)
        # wandb.log({"H clustering": wandb.Image(img_path)})

        # Close the plot display window to conserve system resources.
        # TODO
        # plt.close()
        # wandb.finish()

    def plot_kmeans_cluster_scores(self, embeddings: List, true_labels: List, k_values: List[int], project_name=None,
                                   run_name=None):
        """ This function, using implemented metrics in clustering_metrics, calculates and plots both purity scores and silhouette scores for various numbers of clusters.
        Then using wandb plots the respective scores (each with a different color) for each k value.

        Parameters
        -----------
        embeddings : List
            A list of vectors representing the data points.
        true_labels : List
            A list of ground truth labels for each data point.
        k_values : List[int]
            A list containing the various values of 'k' (number of clusters) for which the scores will be calculated.
            Default is range(2, 9), which means it will calculate scores for k values from 2 to 8.
        project_name : str
            Your wandb project name. If None, the plot will not be logged to wandb. Default is None.
        run_name : str
            Your wandb run name. If None, the plot will not be logged to wandb. Default is None.

        Returns
        --------
        None
        """
        CM = ClusteringMetrics()
        s_scores = []
        p_scores = []
        # Calculating Silhouette Scores and Purity Scores for different values of k
        for k in k_values:
            # TODO
            centers, labels = self.cluster_kmeans(embeddings, k)

            # Using implemented metrics in clustering_metrics, get the score for each k in k-means clustering
            # and visualize it.
            # TODO
            s_scores.append(CM.silhouette_score(embeddings, labels))
            p_scores.append(CM.purity_score(true_labels, labels))

            # Plotting the scores
        # TODO
        plt.figure(figsize=(8, 6))
        plt.plot(k_values, s_scores, label='S score')
        plt.plot(k_values, p_scores, label='P score')
        plt.legend()
        plt.xlabel('n-clusters')
        plt.ylabel('score')
        plt.show()

        # Logging the plot to wandb
        # if project_name and run_name:
        #    import wandb
        #    wandb.login(key="63279235fbf3eb23c36ab8ad68bb00ffae0a06f9")
        #    run = wandb.init(project=project_name, name=run_name)
        #    wandb.log({"Cluster Scores": plt})

        # plt.close()
        # wandb.finish()

    def visualize_elbow_method_wcss(self, embeddings: List, k_values: List[int], project_name: str, run_name: str):
        """ This function implements the elbow method to determine the optimal number of clusters for K-means clustering based on the Within-Cluster Sum of Squares (WCSS).

        The elbow method is a heuristic used to determine the optimal number of clusters in K-means clustering. It involves plotting the WCSS values for different values of K (number of clusters) and finding the "elbow" point in the curve, where the marginal improvement in WCSS starts to diminish. This point is considered as the optimal number of clusters.

        The function performs the following steps:
        1. Iterate over the specified range of K values.
        2. For each K value, perform K-means clustering using the `cluster_kmeans_WCSS` function and store the resulting WCSS value.
        3. Create a line plot of WCSS values against the number of clusters (K).
        4. Log the plot to Weights & Biases (wandb) for visualization and tracking.

        Parameters
        -----------
        embeddings: List
            A list of vectors representing the data points to be clustered.
        k_values: List[int]
            A list of K values (number of clusters) to explore for the elbow method.
        project_name: str
            The name of the wandb project to log the elbow method plot.
        run_name: str
            The name of the wandb run to log the elbow method plot.

        Returns
        --------
        None
        """
        # Initialize wandb
        # wandb.login(key="63279235fbf3eb23c36ab8ad68bb00ffae0a06f9")

        # run = wandb.init(project=project_name, name=run_name)

        # Compute WCSS values for different K values
        wcss_values = []
        for k in k_values:
            # TODO
            wcss_values.append(self.cluster_kmeans_WCSS(embeddings, k)[2])

        # Plot the elbow method
        # TODO
        plt.figure(figsize=(8, 6))
        plt.plot(k_values, wcss_values, label='WCSS')
        plt.xlabel('n-clusters')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')
        plt.show()

        # Log the plot to wandb
        # wandb.log({"Elbow Method": wandb.Image(plt)})

        # plt.close()
        # wandb.finish()


if __name__ == '__main__':
    pass
