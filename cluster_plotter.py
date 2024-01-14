from sklearn.manifold import TSNE
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class ClusterPlotter:

    count = 0

    @staticmethod
    def plot_clusters(labels, query):
        if ClusterPlotter.count % 100 == 0:
            query = query.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(query)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(query)  

            plt.scatter(X_tsne[:,0], X_tsne[:,1], c=labels, s=50)
            plt.savefig(f'cluster_plots/4_clusters_{ClusterPlotter.count}_tsne.png')

            plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, s=50)
            plt.savefig(f'cluster_plots/4_clusters_{ClusterPlotter.count}_pca.png')
        
        ClusterPlotter.count += 1
        pass
