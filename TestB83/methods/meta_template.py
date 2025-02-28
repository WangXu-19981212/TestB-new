import torch.nn as nn
import torch
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from methods.tools import consistency_loss


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, flatten=True, leakyrelu=False, tf_path=None, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1  # (change depends on input)
        self.feature = model_func(flatten=flatten, leakyrelu=leakyrelu)
        self.feat_dim = self.feature.final_feat_dim
        self.change_way = change_way  # some methods allow different_way classification during training and test
        self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        """
        处理输入数据并提取支持集和查询集的特征。

        Args:
            x (torch.Tensor): 输入数据，可以是原始图像数据或已经提取的特征。
            is_feature (bool): 表示输入数据是否已经是特征。

        Returns:
            z_support (torch.Tensor): 支持集的特征。
            z_query (torch.Tensor): 查询集的特征。
        """
        # 将输入数据移动到GPU
        x = x.cuda()

        if is_feature:
            # 如果输入数据已经是特征，则直接使用
            z_all = x
        else:
            # 调整输入数据的形状
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])

            # 通过特征提取器提取特征
            z_all = self.feature.forward(x)

            # 调整特征的形状
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)

        # 提取支持集和查询集的特征
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x):
        scores, loss = self.set_forward_loss(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query), loss.item() * len(y_query)

    def train_loop(self, epoch, train_loader_ori, optimizer, total_it):
        """
        训练循环，更新模型参数并计算损失。

        Args:
            epoch (int): 当前训练的轮数。
            train_loader_ori (DataLoader): 训练数据加载器。
            optimizer (Optimizer): 优化器。
            total_it (int): 总的迭代次数。

        Returns:
            total_it (int): 更新后的总迭代次数。
        """
        print_freq = len(train_loader_ori) // 10
        avg_loss = 0

        for i, (x_ori, global_y) in enumerate(train_loader_ori):
            self.n_query = x_ori.size(1) - self.n_support
            if self.change_way:
                self.n_way = x_ori.size(0)

            optimizer.zero_grad()

            epsilon_list = [0.1, 0.01, 0.001]

            # 计算原始和对抗样本的损失
            scores_fsl_ori, loss_fsl_ori, scores_cls_ori, loss_cls_ori, scores_fsl_adv, loss_fsl_adv, scores_cls_adv, loss_cls_adv = self.set_forward_loss_method(epoch,
                x_ori, global_y, epsilon_list)

            # 计算原始和对抗样本之间的 KL 散度损失
            if scores_fsl_ori.equal(scores_fsl_adv):
                loss_fsl_KL = 0
            else:
                loss_fsl_KL = consistency_loss(scores_fsl_ori, scores_fsl_adv, 'KL3')

            if scores_cls_ori.equal(scores_cls_adv):
                loss_cls_KL = 0
            else:
                loss_cls_KL = consistency_loss(scores_cls_ori, scores_cls_adv, 'KL3')

            # 计算最终的损失
            k1, k2, k3, k4, k5, k6 = 1, 1, 1, 1, 0, 0
            loss = k1 * loss_fsl_ori + k2 * loss_fsl_adv + k3 * loss_fsl_KL + k4 * loss_cls_ori + k5 * loss_cls_adv + k6 * loss_cls_KL

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 更新平均损失
            avg_loss += loss.item()

            # 打印训练信息
            if (i + 1) % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader_ori),
                                                                        avg_loss / float(i + 1)))
                print(
                    'loss_fsl_ori {:f} | loss_fsl_adv {:f} | loss_fsl_KL {:f} | loss_cls_ori {:f}'.format(loss_fsl_ori,
                                                                                                          loss_fsl_adv,
                                                                                                          loss_fsl_KL,
                                                                                                          loss_cls_ori))
            # 记录 TensorBoard 日志
            if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
                self.tf_writer.add_scalar('loss_fsl_ori:', loss_fsl_ori.item(), total_it + 1)
                self.tf_writer.add_scalar('loss_fsl_adv:', loss_fsl_adv.item(), total_it + 1)
                self.tf_writer.add_scalar('loss_cls_ori:', loss_cls_ori.item(), total_it + 1)
                self.tf_writer.add_scalar('loss_cls_adv:', loss_cls_adv.item(), total_it + 1)
                self.tf_writer.add_scalar('total_loss:', loss.item(), total_it + 1)
                self.tf_writer.add_scalar(self.method + '/query_loss', loss.item(), total_it + 1)

            # 更新总迭代次数
            total_it += 1

        return total_it

    def test_loop(self, test_loader, record=None):
        """
        在测试集上评估模型的性能。

        Args:
            test_loader (DataLoader): 测试数据加载器。
            record (dict, optional): 记录测试结果的字典。

        Returns:
            acc_mean (float): 平均准确率。
        """
        loss = 0.
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)

            # 计算当前批次的准确率和损失
            correct_this, count_this, loss_this = self.correct(x)

            # 将当前批次的准确率添加到列表中
            acc_all.append(correct_this / count_this * 100)

            # 更新总损失和总样本数
            loss += loss_this
            count += count_this

        # 计算平均准确率和标准差
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)

        # 打印测试结果
        print('--- %d Loss = %.6f ---' % (iter_num, loss / count))
        print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_mean