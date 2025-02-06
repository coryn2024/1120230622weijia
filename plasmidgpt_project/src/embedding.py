import torch
from transformers import PreTrainedTokenizerFast
import numpy as np
import os
from Bio import SeqIO


def validate_sequence(sequence):
    """
    验证DNA序列是否只包含ATCGN字符
    如果检测到非法字符，将抛出ValueError
    """
    valid_characters = set("ATCGN")
    sequence_upper = sequence.upper()
    if not set(sequence_upper).issubset(valid_characters):
        raise ValueError(f"Invalid character(s) found in sequence: {sequence}")
    return sequence_upper


def load_model(model_path, device):
    """
    加载预训练模型
    model_path: 例如 "pretrained_model/pretrained_model.pt"
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    # 使用 torch.load 加载本地模型
    model = torch.load(model_path)
    model.config.output_hidden_states = True  # 开启输出隐藏状态
    model.eval()
    model = model.to(device)
    return model


def load_tokenizer(tokenizer_file):
    """
    加载分词器
    tokenizer_file: 例如 "pretrained_model/addgene_trained_dna_tokenizer.json"
    """
    if not os.path.exists(tokenizer_file):
        raise FileNotFoundError(f"Tokenizer file '{tokenizer_file}' not found.")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    return tokenizer


def calculate_embeddings(model, tokenizer, sequence, device):
    """
    对单条DNA序列计算嵌入
    使用模型的最后一层隐藏状态，并通过平均池化获取序列级向量
    """
    # 将序列转大写并转换为 input_ids
    input_ids = tokenizer.encode(sequence.upper(), return_tensors='pt',
                                 truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        # 取最后一层 hidden_states
        hidden_states = outputs.hidden_states[-1].cpu().numpy()
        # 对序列长度进行平均，得到序列级别的向量表示
        hidden_states_mean = np.mean(hidden_states, axis=1).reshape(-1)

    return hidden_states_mean


def read_sequences(fasta_file):
    """
    从FASTA文件中读取所有DNA序列
    并使用 validate_sequence() 函数做字符验证
    """
    sequences = []
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"FASTA file '{fasta_file}' not found.")

    with open(fasta_file, 'r') as file:
        for record in SeqIO.parse(file, "fasta"):
            validated_sequence = validate_sequence(str(record.seq))
            sequences.append(validated_sequence)

    return sequences


def compute_embeddings(sequences, model_dir, device):
    """
    对多条序列依次计算嵌入
    model_dir: 存放模型和分词器的目录
    """
    model_path = os.path.join(model_dir, 'pretrained_model.pt')
    tokenizer_path = os.path.join(model_dir, 'addgene_trained_dna_tokenizer.json')

    # 加载模型和分词器
    model = load_model(model_path, device)
    tokenizer = load_tokenizer(tokenizer_path)

    embeddings = []
    for idx, sequence in enumerate(sequences):
        print(f"Processing sequence {idx + 1}/{len(sequences)}")
        embedding = calculate_embeddings(model, tokenizer, sequence, device)
        embeddings.append(embedding)

    return np.array(embeddings)
