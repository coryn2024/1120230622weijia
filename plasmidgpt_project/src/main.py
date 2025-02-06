import numpy as np
import torch

from embedding import read_sequences, compute_embeddings
from tsne_visualization import visualize_tsne

def main():
    # 选择计算设备(GPU或CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 路径设置
    fasta_file = "data/plasmids.fasta"           # FASTA文件路径
    model_dir = "pretrained_model"               # 模型和分词器存放目录
    output_embeddings = "output/embeddings.npy"  # 嵌入保存文件
    output_tsne_plot = "output/tsne_plot.png"    # t-SNE可视化图保存文件

    # 1. 读取DNA序列
    print("Reading sequences...")
    sequences = read_sequences(fasta_file)
    print(f"Total sequences loaded: {len(sequences)}")

    # 2. 计算嵌入
    print("Calculating embeddings...")
    embeddings = compute_embeddings(sequences, model_dir, device)

    # 3. 保存嵌入
    np.save(output_embeddings, embeddings)
    print(f"Embeddings saved to {output_embeddings}")

    # 4. t-SNE 可视化
    print("Generating t-SNE visualization...")
    visualize_tsne(embeddings, output_tsne_plot)
    print(f"t-SNE plot saved to {output_tsne_plot}")

if __name__ == "__main__":
    main()
