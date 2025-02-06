import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_tsne(embeddings, output_file):
    """
    将嵌入进行t-SNE降维，并保存可视化图
    """
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    plt.title("t-SNE Visualization of DNA Sequence Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    plt.savefig(output_file)
    plt.show()
