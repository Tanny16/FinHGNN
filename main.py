import pickle
from torch import optim
import argparse
import os
import pandas as pd
from Model import *

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default='5e-4',
                    help='Learning rate ')
parser.add_argument('--rnn_hidden_size', type=int, default='256',
                    help='rnn hidden nodes')
parser.add_argument('--GNN_hidden_size', type=int, default='128',
                    help='gnn hidden nodes')
parser.add_argument('--rnn-length', type=int, default='30',
                    help='rnn length')
parser.add_argument('--dropout', type=float, default='0.2',
                    help='dropout rate')
parser.add_argument('--clip', type=float, default='0.0025',
                    help='rnn clip')
parser.add_argument('--relation', type=str, default='attention_gate'
                    )
parser.add_argument('--save', type=bool, default=True,
                    help='save model')


def load_dataset(DEVICE, num):
    with open('./final_data/Feature_normalized_sample10.pkl', 'rb') as handle:
        stock_feature = pickle.load(handle)
        stock_feature = stock_feature[:-1, :, :5]
        stock_feature.to(torch.float)

    with open('./final_data/y_pickle_sample10.pkl', 'rb') as handle:
        y_load = pickle.load(handle)
    with open('./final_data/topicInfo/topicInfo_' + str(num) + '.pkl', 'rb') as handle:
        topicInfo = pickle.load(handle)
        topicInfo = topicInfo[:-1, :, :]
    with open('./final_data/relation_static1/relation_sample10.pkl', 'rb') as handle:
        relation_static = pickle.load(handle)
        relation_static = relation_static[:-1, :, :]
    with open('./final_data/adj_analytic_aa_total.pkl', 'rb') as handle:
        adj_analytic_aa_total = pickle.load(handle)
    with open('./final_data/adj_analytic_sa_total_sample10.pkl', 'rb') as handle:
        adj_analytic_sa_total = pickle.load(handle)
    with open('./final_data/analyst.pkl', 'rb') as handle:
        analystInfo = pickle.load(handle)
    print('DEVICE:', DEVICE)
    stock_feature = stock_feature.to(DEVICE)
    y_load = y_load.to(DEVICE)
    relation_static = relation_static.to(DEVICE)
    topicInfo = topicInfo.to(DEVICE)
    adj_analytic_sa_total = adj_analytic_sa_total.to(DEVICE)
    adj_analytic_aa_total = adj_analytic_aa_total.to(DEVICE)
    analystInfo = analystInfo.to(DEVICE)
    return stock_feature, y_load, topicInfo, relation_static, adj_analytic_sa_total, adj_analytic_aa_total, analystInfo


def train(model, x_train, train_y, tensor_adjacency, rnn_length_x, x_topicInfo,adj_analytic_sa, adj_analytic_aa,analystInfo):
    model.train()
    seq_len = len(x_train)
    train_seq = list(range(seq_len))[rnn_length_x:]
    total_loss = 0
    total_loss_count = 0
    batch_train = 16  # batch_size的设置

    for i in train_seq:
        x_train_i = x_train[:][i - rnn_length_x:i].to(device)
        tensor_adjacency_i = tensor_adjacency[i - 1].to(device)
        x_topicInfo_train_i = x_topicInfo[i - 1].to(device)

        output, weight = model(tensor_adjacency_i, x_train_i, x_topicInfo_train_i,adj_analytic_sa, adj_analytic_aa,analystInfo)
        loss = criterion(output, train_y[i - 1].long().to(device))

        loss.backward()

        total_loss += loss.item()
        total_loss_count += 1
        if total_loss_count % batch_train == batch_train - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
    if total_loss_count % batch_train != batch_train - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
    torch.cuda.empty_cache()

    return total_loss / total_loss_count


def evaluate(model, x_eval, y_eval, tensor_adjacency, rnn_length_x, x_topicInfo,adj_analytic_sa,adj_analytic_aa,analystInfo):
    model.eval()
    seq_len = len(x_eval)
    seq = list(range(seq_len))[rnn_length_x:]
    preds = []
    trues = []
    preds_possible = -torch.ones(len(seq), x_eval.shape[1], 2)
    preds_save = -torch.ones(len(seq), x_eval.shape[1])
    trues_save = -torch.ones(len(seq), x_eval.shape[1])
    total_loss = 0
    total_loss_count = 0
    for i in seq:
        x_eval_i = x_eval[:][i - rnn_length_x:i].to(device)
        tensor_adjacency_i = tensor_adjacency[i - 1].to(device)  # 5~49
        x_topicInfo_train_i = x_topicInfo[i - 1].to(device)
        with torch.no_grad():
            output, weight = model(tensor_adjacency_i, x_eval_i, x_topicInfo_train_i,adj_analytic_sa, adj_analytic_aa,analystInfo)
        loss = criterion(output, y_eval[i - 1].long().to(device)).to(device)
        total_loss += loss.item()
        total_loss_count += 1
        output = output.detach().cpu()
        preds_save[i - rnn_length_x] = output.argmax(-1)
        preds_possible[i - rnn_length_x] = output
        trues_save[i - rnn_length_x] = y_eval[i - 1].long()
        preds.append(np.exp(output.numpy()))
        trues.append(y_eval[i - 1].cpu().numpy())
    acc, auc, mcc = metrics1(trues, preds)
    return {"acc": acc, "auc": auc, "mcc": mcc,
            "loss": total_loss / total_loss_count}, preds_save, trues_save, preds_possible, weight


def save_preds(best_preds, best_trues, best_preds_possible, file_name='eval'):
    cl_file = "./model_result/HGCN_result_pred_dynamic_attention_gate/" + file_name + ".pkl"  # 保存预测结果和概率
    output = open(cl_file, 'wb')
    pickle.dump((best_preds, best_trues, best_preds_possible), output)
    output.close()


def save_record(dicts, seed, file_name='train'):
    dicts.to_excel('./model_result/HGCN_result_loss_dynamic_attention_gate/' + file_name + '__' + str(
        seed) + '.xlsx')  # 保存train_loss_history,eval_acc_history,eval_auc_history。


if __name__ == '__main__':
    args = parser.parse_args(args=[])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(0)
    print(device)

    lr_list = [ 0.0001,0.00005]
    days_list = [40]

    for num in days_list:
        stock_feature, y, topicInfo, relation_static, adj_analytic_sa, adj_analytic_aa, analystInfo = load_dataset(device, num)
        global num_stock
        global num_topic
        num_stock = stock_feature.shape[1]
        num_topic = topicInfo.shape[1]
        num_analyst = analystInfo.shape[0]
        for lrate in lr_list:

                for i in range(0, 3):  # 一次循环训练一个模型
                    seed = random.randint(1, 100000)
                    set_seed(seed)
                    relation = args.relation
                    args.rnn_hidden_size = 32
                    args.GNN_hidden_size = 64
                    rnn_hidden_size = args.rnn_hidden_size
                    GNN_hidden_size = args.GNN_hidden_size

                    model = GcnNet(d_markets=stock_feature.shape[-1],
                                   rnn_hidden_size=rnn_hidden_size,
                                   GNN_hidden_size=GNN_hidden_size,
                                   num_stock=num_stock,
                                   num_topic=num_topic,
                                   num_analyst=num_analyst
                                   )
                    model.to(device)

                    print("model:", model.state_dict().keys())
                    optimizer = optim.Adam(model.parameters(), lr=lrate)  # 0.00002
                    criterion = nn.CrossEntropyLoss()  # 多分类 （负面、正面、中性）

                    rnn_length_x = 20  # 以前t天的基本面信息得到当天的股票节点特征
                    days = stock_feature.shape[0]

                    x_train = stock_feature[:int(days / 5 * 4)]
                    x_eval = stock_feature[int(days / 5 * 4) - rnn_length_x: int(days / 10 * 9)]  # int(days/4*3)
                    x_test = stock_feature[int(days / 10 * 9) - rnn_length_x:]

                    y_train = y[:int(days / 5 * 4)]
                    y_eval = y[int(days / 5 * 4) - rnn_length_x:int(days / 10 * 9)]
                    y_test = y[int(days / 10 * 9) - rnn_length_x:]

                    x_topicInfo_train = topicInfo[:int(days / 5 * 4)]
                    x_topicInfo_eval = topicInfo[int(days / 5 * 4) - rnn_length_x: int(days / 10 * 9)]
                    x_topicInfo_test = topicInfo[int(days / 10 * 9) - rnn_length_x:]

                    relation_static_train = relation_static[:int(days / 5 * 4)]
                    relation_static_eval = relation_static[int(days / 5 * 4) - rnn_length_x: int(days / 10 * 9)]
                    relation_static_test = relation_static[int(days / 10 * 9) - rnn_length_x:]

                    MAX_EPOCH = 300
                    best_model_file = 0
                    epoch = 0
                    wait_epoch = 0
                    eval_epoch_best = 0

                    eval_df = pd.DataFrame()
                    test_df = pd.DataFrame()
                    train_loss_history = pd.DataFrame()
                    train_loss_history1 = []
                    while epoch < MAX_EPOCH:
                        train_loss = train(model,
                                           x_train,
                                           y_train,
                                           relation_static_train,
                                           rnn_length_x,
                                           x_topicInfo_train,
                                           adj_analytic_sa,
                                           adj_analytic_aa,
                                           analystInfo
                                           )
                        train_loss_history1.append(train_loss)
                        eval_dict, preds_save, trues_save, preds_possible, weight = evaluate(model,
                                                                                             x_eval,
                                                                                             y_eval,
                                                                                             relation_static_eval,
                                                                                             rnn_length_x,
                                                                                             x_topicInfo_eval,
                                                                                             adj_analytic_sa,
                                                                                             adj_analytic_aa,
                                                                                             analystInfo
                                                                                             )
                        eval_df = eval_df.append(eval_dict, ignore_index=True)
                        test_dict, preds_save_test, trues_save_test, preds_possible_test, weight_test = evaluate(model,
                                                                                                                 x_test,
                                                                                                                 y_test,
                                                                                                                 relation_static_test,
                                                                                                                 rnn_length_x,
                                                                                                                 x_topicInfo_test,
                                                                                                                 adj_analytic_sa,
                                                                                                                 adj_analytic_aa,
                                                                                                                 analystInfo
                                                                                                                 )
                        test_df = test_df.append(test_dict, ignore_index=True)
                        eval_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f}," \
                                   "test_acc{:.4f}".format(
                            epoch, train_loss, eval_dict['auc'], eval_dict['acc'], test_dict['auc'], test_dict['acc'])
                        print(eval_str)

                        if (eval_dict['auc'] > eval_epoch_best):
                            eval_epoch_best = eval_dict['auc']
                            eval_best_str = "relation{},epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, " \
                                            "test_auc{:.4f},test_acc{:.4f}".format(
                                relation, epoch, train_loss, eval_dict['auc'], eval_dict['acc'], test_dict['auc'],
                                test_dict['acc'])
                            wait_epoch = 0
                            if args.save:
                                if best_model_file:
                                    os.remove(best_model_file)
                                    os.remove(best_record_file1)
                                    os.remove(best_record_file2)
                                best_model_file = "./SavedModels/dynamic_{}_num{}_rnn{}_gnn{}_seed{}_lr{" \
                                                  "}_eval,auc{:.4f}_acc{:.4f}_test,auc{:.4f}_acc{:.4f}_epoch{" \
                                                  "}.pkl".format(
                                    relation, num, args.rnn_hidden_size, args.GNN_hidden_size, seed, lrate,
                                    eval_dict['auc'], eval_dict['acc'], test_dict['auc'], test_dict['acc'], epoch, )
                                torch.save(model.state_dict(), best_model_file)
                                best_model_file_weight = "./model_result/weight/dynamic_{}_num{}_rnn{}_gnn{}_seed{" \
                                                         "}_lr{}_eval,auc{:.4f}_acc{:.4f}_test,auc{:.4f}_acc{:.4f}_epoch{" \
                                                         "}.pkl".format(
                                    relation, num, args.rnn_hidden_size, args.GNN_hidden_size, seed, lrate,
                                    eval_dict['auc'], eval_dict['acc'], test_dict['auc'], test_dict['acc'], epoch)
                                torch.save(weight_test, best_model_file_weight)
                                eval_name = relation + "_" + str(lrate) + "_" + str(
                                    num) + ",seed{},eval,auc{:.4f}_acc{:.4f}_mcc{:.4f}_epoch{}_".format(
                                    seed, eval_dict['auc'], eval_dict['acc'], eval_dict['mcc'], epoch)
                                test_name = relation + "_" + str(lrate) + "_" + str(
                                    num) + ",seed{},test,auc{:.4f}_acc{:.4f}_mcc{:.4f}_epoch{}_".format(
                                    seed, test_dict['auc'], test_dict['acc'], test_dict['mcc'], epoch)
                                best_record_file1 = "./model_result/HGCN_result_pred_dynamic_attention_gate/" + eval_name + ".pkl"
                                best_record_file2 = "./model_result/HGCN_result_pred_dynamic_attention_gate/" + test_name + ".pkl"
                                save_preds(preds_save, trues_save, preds_possible, eval_name)
                                save_preds(preds_save_test, trues_save_test, preds_possible_test, test_name)
                        else:
                            wait_epoch += 1

                        if wait_epoch > 30:
                            print("saved_model_result:", eval_best_str)
                            break
                        epoch += 1
                    train_loss_history["train_loss_history"] = train_loss_history1
                    save_record(train_loss_history, seed, 'train')
