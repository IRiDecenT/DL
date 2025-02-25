import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

# 两层GCN
class GCN(nn.Module):
    # nfeat=1433 nhid=16 nclass=7 dropout=0.5
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # x 2708 * 1433 -> 2708 * 16 -> 2708 * 7
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # 最后一层激活函数选用log_softmax 输出是每个节点属于不同类别的对数概率，这些对数概率可以用于后续的分类任务。
        return F.log_softmax(x, dim=1)
