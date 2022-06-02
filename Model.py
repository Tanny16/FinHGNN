from Layers import *


class GcnNet(nn.Module):  ##整个模型
    """
    定义一个包含两层GraphConvolution的模型
    """

    def __init__(self, d_markets, rnn_hidden_size, GNN_hidden_size, num_stock, num_topic,num_analyst, dropout=0.2):
        super(GcnNet, self).__init__()
        self.dropout = dropout
        self.lstm = simple_LSTM(hidden_size=rnn_hidden_size,
                                embedding_dim=d_markets)
        self.gcn1 = GraphConvolution(d_markets=d_markets,
                                     rnn_hidden_size=rnn_hidden_size,
                                     GNN_hidden_size=GNN_hidden_size,
                                     GNN_output_size=GNN_hidden_size,
                                     num_stock=num_stock,
                                     num_topic=num_topic,
                                     num_analyst=num_analyst)

        self.wMatrix_pred = nn.Parameter(torch.Tensor(GNN_hidden_size,
                                                      GNN_hidden_size).requires_grad_(True))
        self.bias = nn.Parameter(torch.Tensor(GNN_hidden_size).requires_grad_(True))
        self.fc1 = fc_layer(GNN_hidden_size, 2)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, tensor_adjacency_i, x_train_i, x_topicInfo_train_i,adj_analytic_sa, adj_analytic_aa,analystInfo):
        stock_feature = self.lstm(x_train_i)
        topic_feature = F.dropout(x_topicInfo_train_i, self.dropout, training=self.training)
        stock_feature = F.dropout(stock_feature, self.dropout, training=self.training)
        h, weight = self.gcn1(tensor_adjacency_i, stock_feature, topic_feature,adj_analytic_sa, adj_analytic_aa, analystInfo)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.elu(self.fc1(h[:1048]))
        outcome = F.log_softmax(h[:1048], dim=1)
        return outcome, weight
