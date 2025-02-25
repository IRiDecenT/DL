import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    # 2708 * 1435  行向量：节点编号 特征... 论文类别（Neural_Networks等） 
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # 提取第二到倒数第二列所有特征 转化为稀疏矩阵
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    
    # 提取最后一列的标签并 onehot编码 2708(节点数) * 7（论文种类）
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # 节点编号数组
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 形成节点编号与其对应在idx索引中的映射 key：节点编号 value：编号在数组中的索引
    idx_map = {j: i for i, j in enumerate(idx)}
    # (5429*2) 节点编号 —— 节点编号 形成的引用关系
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # (5429*2) 节点编号对应索引 —— 节点编号对应索引
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 创建稀疏的COO（coordinate List）格式的邻接矩阵adj
    # 第一个参数指定矩阵非零元素的坐标
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    # 注：在数据集文件cora.cites中给出边是有向的，而在引文网络中我们认为是无向边，所以该邻接矩阵需要转化为对称的邻接矩阵
    # 将有向图转化为无向图
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 对特征归一化
    features = normalize(features)
    # 对A+I归一化  sp.eye(adj.shape[0])即为I 稀疏单位矩阵。单位矩阵的对角线上全是1，其他位置全是0。
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # 将numpy格式转化为torch中的tensor格式
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))       # 矩阵行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.    # 如果是inf则转化为0
    r_mat_inv = sp.diags(r_inv)    # 构造对角矩阵
    mx = r_mat_inv.dot(mx)         # 按行归一化  D^-1 * A
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
