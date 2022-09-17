from torch import tensor, float as t_float, long
from torch.nn import Embedding
from pandas import DataFrame as df
from numpy import array as arr
from matplotlib import pyplot as plt
from networkx import draw_networkx, spring_layout
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

# Helper function for visualization.

def visualize_graph(G):
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    draw_networkx(G, pos=spring_layout(G, seed=42), with_labels=True, cmap="Set2", node_size=1000)
    plt.show()

def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()

# Untuk calculate correlation
# ==================================================


def calc_corr(input_data):
    input_data = df(input_data)
    return input_data.corr(method='pearson') # calculate corr between SNPs

# Untuk filter data (ubah data < threshold dan == 1 jadi string 'Invalid')
# ==================================================


def filter_data(input_data, threshold_value=0.07):
    input_data = arr(input_data)

    valid_corr_vals0 = []
    valid_corr_vals1 = []

    for i in range(0, len(input_data)):
        for j in range(0, len(input_data[i])):
            if input_data[i][j] < threshold_value or input_data[i][j] == 1:
                valid_corr_vals1.append('Invalid')
            else:
                valid_corr_vals1.append(input_data[i][j])
        valid_corr_vals0.append(valid_corr_vals1)
        valid_corr_vals1 = []

    return valid_corr_vals0

# EDGE INDEX WITHOUT MIRROR
# ===================================================


def crt_edge_index(input_data):
    edge_index_array1 = []
    edge_index_array2 = []
    saved_index = []

    for i in range(0, len(input_data)):
        for j in range(i+1, len(input_data[i])):
            if input_data[i][j] != 'Invalid':
                edge_index_array1.append(i)
                edge_index_array2.append(j)

    # valid_vertices = get_valid_vertices(edge_index_array1, edge_index_array2)
    edge_index = tensor([edge_index_array1, edge_index_array2], dtype=long)
    return edge_index

def get_graph_data(section_x, section_y, embedding_dim):
    
    transposed_data_x =  section_x.transpose()
    corr_vals = calc_corr(transposed_data_x)

    # Filter data berdasarkan threshold correlation value
    corr_vals_np = arr(corr_vals)
    valid_corr_val = filter_data(corr_vals_np)

    # Hitung jumlah valid data
    valid_corr_val = arr(valid_corr_val)

    invalid_data = 0
    valid_data = 0

    for i in valid_corr_val:
        for j in i:
            if j == 'Invalid':
                invalid_data += 1
            else:
                valid_data += 1
    total_data = invalid_data + valid_data

    num_valid_data = [valid_data, invalid_data, total_data]

    # create edge index
    edge_index = []
    edge_index = crt_edge_index(valid_corr_val)

    # get embedding
    section_x = arr(section_x)
    section_y = arr(section_y)

    section_x = tensor(section_x, dtype=long)
    section_y = tensor(section_y, dtype=long)

    embedding = Embedding(3, embedding_dim)
    embedd_x = embedding(section_x)

    data_x = tensor(embedd_x, dtype=t_float)
    data_y = tensor(section_y, dtype=t_float)

    # membuat Data dengan parameter: x, y, edge_index
    
    graph_data = Data(x=data_x, y=data_y, edge_index=edge_index)
    G_graph = to_networkx(graph_data, to_undirected=True)
    # visualize_graph(G)

    return graph_data, G_graph, num_valid_data