import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import tanh
from utils import *


def attribute_aware(self, stock_feature, topic_feature, analyst_feature):
    transform_1 = self.w_attribute_1(torch.transpose(stock_feature, 0, 1).unsqueeze(dim=0))
    transform_2 = self.w_attribute_2(torch.transpose(topic_feature, 0, 1).unsqueeze(dim=0))
    transform_3 = self.w_attribute_3(torch.transpose(analyst_feature, 0, 1).unsqueeze(dim=0))
    transform_1 = torch.transpose(transform_1.squeeze(0), 0, 1)
    transform_2 = torch.transpose(transform_2.squeeze(0), 0, 1)
    transform_3 = torch.transpose(transform_3.squeeze(0), 0, 1)
    gate_ss = F.elu(transform_1.unsqueeze(1) + transform_1)
    gate_st = F.elu(transform_1.unsqueeze(1) + transform_2)
    gate_ts = F.elu(transform_2.unsqueeze(1) + transform_1)
    gate_tt = F.elu(transform_2.unsqueeze(1) + transform_2)
    gate_as = F.elu(transform_3.unsqueeze(1) + transform_1)
    gate_sa = F.elu(transform_1.unsqueeze(1) + transform_3)
    gate_aa = F.elu(transform_3.unsqueeze(1) + transform_3)
    return {'gate_ss': gate_ss, 'gate_st': gate_st, 'gate_ts': gate_ts, 'gate_tt': gate_tt, 'gate_sa': gate_sa,
            'gate_aa': gate_aa, 'gate_as': gate_as}


def type_aware(self, stock_feature, topic_feature, analyst_feature):
    weight_ss = torch.zeros(self.num_stock, self.num_stock, device=stock_feature.device, dtype=stock_feature.dtype)
    f_1 = self.alphaVector_ss_1(torch.transpose(stock_feature, 0, 1).unsqueeze(0))
    f_2 = self.alphaVector_ss_2(torch.transpose(stock_feature, 0, 1).unsqueeze(0))
    weight_ss += (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)
    weight_ss = F.elu(weight_ss)
    temp_index = torch.nonzero(self.weight_adjacency_ss == 0, as_tuple=True)
    weight_ss[temp_index[0], temp_index[1]] = -11111
    weight_ss = F.softmax(weight_ss, dim=1)

    weight_st = torch.zeros(self.num_stock, self.num_topic, device=stock_feature.device, dtype=stock_feature.dtype)
    f_1 = self.alphaVector_st_1(torch.transpose(stock_feature, 0, 1).unsqueeze(0))
    f_2 = self.alphaVector_st_2(torch.transpose(topic_feature, 0, 1).unsqueeze(0))
    weight_st += (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)
    weight_st = F.elu(weight_st)
    temp_index = torch.nonzero(self.weight_adjacency_st == 0, as_tuple=True)
    weight_st[temp_index[0], temp_index[1]] = -11111
    weight_st = F.softmax(weight_st, dim=1)

    weight_ts = torch.zeros(self.num_topic, self.num_stock, device=stock_feature.device, dtype=stock_feature.dtype)
    f_1 = self.alphaVector_ts_1(torch.transpose(topic_feature, 0, 1).unsqueeze(0))
    f_2 = self.alphaVector_ts_2(torch.transpose(stock_feature, 0, 1).unsqueeze(0))
    weight_ts += (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)
    weight_ts = F.elu(weight_ts)
    temp_index = torch.nonzero(self.weight_adjacency_ts == 0, as_tuple=True)
    weight_ts[temp_index[0], temp_index[1]] = -11111
    weight_ts = F.softmax(weight_ts, dim=1)

    weight_tt = torch.zeros(self.num_topic, self.num_topic, device=stock_feature.device, dtype=stock_feature.dtype)
    f_1 = self.alphaVector_tt_1(torch.transpose(topic_feature, 0, 1).unsqueeze(0))
    f_2 = self.alphaVector_tt_2(torch.transpose(topic_feature, 0, 1).unsqueeze(0))
    weight_tt += (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)
    weight_tt = F.elu(weight_tt)
    temp_index = torch.nonzero(self.weight_adjacency_tt == 0, as_tuple=True)
    weight_tt[temp_index[0], temp_index[1]] = -11111
    weight_tt = F.softmax(weight_tt, dim=1)
    weight_as = torch.zeros(self.num_analyst, self.num_stock, device=stock_feature.device, dtype=stock_feature.dtype)
    f_1 = self.alphaVector_as_1(torch.transpose(analyst_feature, 0, 1).unsqueeze(0))
    f_2 = self.alphaVector_as_2(torch.transpose(stock_feature, 0, 1).unsqueeze(0))

    weight_as += (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)
    weight_as = F.elu(weight_as)
    temp_index = torch.nonzero(self.weight_adjacency_as == 0, as_tuple=True)
    weight_as[temp_index[0], temp_index[1]] = -11111
    weight_as = F.softmax(weight_as, dim=1)

    weight_sa = torch.zeros(self.num_stock, self.num_analyst, device=stock_feature.device, dtype=stock_feature.dtype)
    f_1 = self.alphaVector_sa_1(torch.transpose(stock_feature, 0, 1).unsqueeze(0))
    f_2 = self.alphaVector_sa_2(torch.transpose(analyst_feature, 0, 1).unsqueeze(0))
    weight_sa += (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)
    weight_sa = F.elu(weight_sa)
    temp_index = torch.nonzero(self.weight_adjacency_sa == 0, as_tuple=True)
    weight_sa[temp_index[0], temp_index[1]] = -11111
    weight_sa = F.softmax(weight_sa, dim=1)

    weight_aa = torch.zeros(self.num_analyst, self.num_analyst, device=stock_feature.device, dtype=stock_feature.dtype)
    f_1 = self.alphaVector_aa_1(torch.transpose(analyst_feature, 0, 1).unsqueeze(0))
    f_2 = self.alphaVector_aa_2(torch.transpose(analyst_feature, 0, 1).unsqueeze(0))
    weight_aa += (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)
    weight_aa = F.elu(weight_aa)
    temp_index = torch.nonzero(self.weight_adjacency_aa == 0, as_tuple=True)
    weight_aa[temp_index[0], temp_index[1]] = -11111
    weight_aa = F.softmax(weight_aa, dim=1)

    return {'weight_ts': weight_ts, 'weight_sa': weight_sa, 'weight_ss': weight_ss, 'weight_st': weight_st,
            'weight_tt': weight_tt, 'weight_aa': weight_aa, 'weight_as': weight_as}


def propregate(self, weight_adjacency, weight, stock_feature, topic_feature, analytic_feature,
               menkong_adjacency):
    weight_ss = torch.mul(weight['weight_ss'], weight_adjacency[:self.num_stock, :self.num_stock])
    weight_st = torch.mul(weight['weight_st'], weight_adjacency[:self.num_stock, self.num_stock:])
    weight_ts = torch.mul(weight['weight_ts'], weight_adjacency[self.num_stock:, :self.num_stock])
    weight_tt = torch.mul(weight['weight_tt'], weight_adjacency[self.num_stock:, self.num_stock:])
    weight_sa = torch.mul(weight['weight_sa'], self.weight_adjacency_sa)
    weight_as = torch.mul(weight['weight_as'], self.weight_adjacency_as)
    weight_aa = torch.mul(weight['weight_aa'], self.weight_adjacency_aa)
    weightss = torch.mul(weight_ss.unsqueeze(dim=2), menkong_adjacency['gate_ss'])
    weightst = torch.mul(weight_st.unsqueeze(dim=2), menkong_adjacency['gate_st'])
    weightts = torch.mul(weight_ts.unsqueeze(dim=2), menkong_adjacency['gate_ts'])
    weighttt = torch.mul(weight_tt.unsqueeze(dim=2), menkong_adjacency['gate_tt'])
    weightsa = torch.mul(weight_sa.unsqueeze(dim=2), menkong_adjacency['gate_sa'])
    weightas = torch.mul(weight_as.unsqueeze(dim=2), menkong_adjacency['gate_as'])
    weightaa = torch.mul(weight_aa.unsqueeze(dim=2), menkong_adjacency['gate_aa'])
    stock_feature_1 = torch.sum(torch.mul(weightss, stock_feature.unsqueeze(dim=0)), dim=1)
    stock_feature_2 = torch.sum(torch.mul(weightst, topic_feature.unsqueeze(dim=0)), dim=1)
    stock_feature_3 = torch.sum(torch.mul(weightsa, analytic_feature.unsqueeze(dim=0)), dim=1)
    topic_feature_1 = torch.sum(torch.mul(weighttt, topic_feature.unsqueeze(dim=0)), dim=1)
    topic_feature_2 = torch.sum(torch.mul(weightts, stock_feature.unsqueeze(dim=0)), dim=1)
    analytic_feature_1 = torch.sum(torch.mul(weightaa, analytic_feature.unsqueeze(dim=0)), dim=1)
    analytic_feature_2 = torch.sum(torch.mul(weightas, stock_feature.unsqueeze(dim=0)), dim=1)
    temp_feature_stock = stock_feature_1 + stock_feature_2 + stock_feature_3
    temp_feature_topic = topic_feature_1 + topic_feature_2
    temp_feature_analytic = analytic_feature_1 + analytic_feature_2

    temp_feature_stock_1 = temp_feature_stock + stock_feature
    return temp_feature_stock_1, temp_feature_topic, temp_feature_analytic


class GraphConvolution(nn.Module):  ##图卷积层
    def __init__(self, d_markets, rnn_hidden_size, GNN_hidden_size, GNN_output_size, num_stock, num_topic, num_analyst,
                 use_bias=True):
        super(GraphConvolution, self).__init__()  # 调用父类的构造方法
        self.use_bias = use_bias
        self.num_stock = num_stock
        self.num_topic = num_topic
        self.num_analyst = num_analyst
        self.wMatrix_sl = nn.Parameter(
            torch.Tensor(rnn_hidden_size, GNN_hidden_size).requires_grad_(True))  # 8为股票节点特征维度
        self.wMatrix_pl = nn.Parameter(torch.Tensor(768, GNN_hidden_size).requires_grad_(True))  # 768为主题节点特征维度
        self.wMatrix_al = nn.Parameter(torch.Tensor(768, GNN_hidden_size).requires_grad_(True))  # 768为主题节点特征维度
        self.w_attribute_1 = nn.Conv1d(GNN_hidden_size, GNN_output_size, kernel_size=1, stride=1)
        self.w_attribute_2 = nn.Conv1d(GNN_hidden_size, GNN_output_size, kernel_size=1, stride=1)
        self.w_attribute_3 = nn.Conv1d(GNN_hidden_size, GNN_output_size, kernel_size=1, stride=1)
        self.alphaVector_ss_1 = nn.Conv1d(GNN_hidden_size, 1, kernel_size=1, stride=1)
        self.alphaVector_ss_2 = nn.Conv1d(GNN_hidden_size, 1, kernel_size=1, stride=1)
        self.alphaVector_st_1 = nn.Conv1d(GNN_hidden_size, 1, kernel_size=1, stride=1)
        self.alphaVector_st_2 = nn.Conv1d(GNN_hidden_size, 1, kernel_size=1, stride=1)
        self.alphaVector_ts_1 = nn.Conv1d(GNN_hidden_size, 1, kernel_size=1, stride=1)
        self.alphaVector_ts_2 = nn.Conv1d(GNN_hidden_size, 1, kernel_size=1, stride=1)
        self.alphaVector_tt_1 = nn.Conv1d(GNN_hidden_size, 1, kernel_size=1, stride=1)
        self.alphaVector_tt_2 = nn.Conv1d(GNN_hidden_size, 1, kernel_size=1, stride=1)
        self.alphaVector_sa_1 = nn.Conv1d(GNN_hidden_size, 1, kernel_size=1, stride=1)
        self.alphaVector_sa_2 = nn.Conv1d(GNN_hidden_size, 1, kernel_size=1, stride=1)
        self.alphaVector_as_1 = nn.Conv1d(GNN_hidden_size, 1, kernel_size=1, stride=1)
        self.alphaVector_as_2 = nn.Conv1d(GNN_hidden_size, 1, kernel_size=1, stride=1)
        self.alphaVector_aa_1 = nn.Conv1d(GNN_hidden_size, 1, kernel_size=1, stride=1)
        self.alphaVector_aa_2 = nn.Conv1d(GNN_hidden_size, 1, kernel_size=1, stride=1)
        self.coef_revise = False
        if self.use_bias:
            self.bias1 = nn.Parameter(torch.Tensor(GNN_hidden_size))
            self.bias2 = nn.Parameter(torch.Tensor(GNN_output_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_bias:
            init.zeros_(self.bias1)
            init.zeros_(self.bias2)

    def forward(self, tensor_adjacency_i, stock_feature, topic_feature, adj_analytic_sa, adj_analytic_aa, analystInfo):

        # 按链接类型分类，归一化构图时的链接权重
        weight_adjacency = tensor_adjacency_i  # 未归一化之前的传播权重
        self.weight_adjacency_ss = weight_adjacency[:self.num_stock, :self.num_stock]
        self.weight_adjacency_ts = weight_adjacency[self.num_stock:, :self.num_stock]
        self.weight_adjacency_st = weight_adjacency[:self.num_stock, self.num_stock:]
        self.weight_adjacency_tt = weight_adjacency[self.num_stock:, self.num_stock:]

        self.weight_adjacency_as = adj_analytic_sa.squeeze(0).t()
        self.weight_adjacency_sa = adj_analytic_sa.squeeze(0)
        self.weight_adjacency_aa = adj_analytic_aa.squeeze(0)

        stock_feature = torch.mm(stock_feature.squeeze(dim=0), self.wMatrix_sl)
        topic_feature = torch.mm(topic_feature, self.wMatrix_pl)
        analyst_feature = torch.mm(analystInfo, self.wMatrix_al)
        # attribute_aware

        menkong_adjacency = attribute_aware(self, stock_feature, topic_feature, analyst_feature)
        # type_aware

        weight = type_aware(self, stock_feature, topic_feature, analyst_feature)

        final_stock_feature, final_topic_feature, final_analyst_feature = propregate(self, weight_adjacency, weight,
                                                                                     stock_feature,
                                                                                     topic_feature, analyst_feature,
                                                                                     menkong_adjacency)  # 一阶传播
        final_stock_feature = tanh(final_stock_feature + self.bias1)

        final_stock_feature, final_topic_feature, final_analyst_feature = propregate(self, weight_adjacency, weight,
                                                                                     final_stock_feature,
                                                                                     final_topic_feature,
                                                                                     final_analyst_feature,
                                                                                     menkong_adjacency)  # 一阶传播
        final_stock_feature = tanh(final_stock_feature + self.bias2)

        return final_stock_feature, weight


class fc_layer(nn.Module):

    def __init__(self, embedding_dim, out_size):
        super(fc_layer, self).__init__()
        self.predictor = nn.Linear(embedding_dim, out_size)

    def forward(self, input_feature):
        pre = self.predictor(input_feature)
        return pre


class simple_LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super(simple_LSTM, self).__init__()
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1)

    def forward(self, seq):
        output, (hidden, cell) = self.encoder(seq.to(torch.float32))
        return hidden
