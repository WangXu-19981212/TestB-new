import numpy as np
import torch
import torch.optim
import os
import random
import time
import sys  # 新增sys模块导入

from methods.backbone_multiblock import model_dict
from data.datamgr import SetDataManager
from methods.main_method import MainMethod

from options import parse_args, get_resume_file, load_warmup_state
from torch.optim.lr_scheduler import StepLR


def print_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Gradient: {param.grad}")


class EarlyStopping:
    def __init__(self, patience=20, delta=0, verbose=False):
        """
        Args:
            patience (int): 允许验证集性能不提升的轮数。
            delta (float): 认为性能提升的最小变化量。
            verbose (bool): 是否打印日志。
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model, checkpoint_dir, epoch):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model, checkpoint_dir, epoch)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model, checkpoint_dir, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, checkpoint_dir, epoch):
        """保存验证集性能最好的模型权重。"""
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save({'epoch': epoch, 'state': model.state_dict()}, os.path.join(checkpoint_dir, 'best_model.tar'))


def train(base_loader, val_loader, model, start_epoch, stop_epoch, params):
    # get optimizer and checkpoint path
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # for validation
    max_acc = 0
    total_it = 0
    # 初始化早停
    early_stopping = EarlyStopping(patience=20, verbose=True)

    # start
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        total_it = model.train_loop(epoch, base_loader, optimizer,
                                    total_it)  # model are called by reference, no need to return
        # print_gradients(model)  # 打印梯度值
        model.eval()

        acc = model.test_loop(val_loader)
        if acc > max_acc:
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
        else:
            print("GG! best accuracy {:f}".format(max_acc))

        # 早停判断
        early_stopping(-acc, model, params.checkpoint_dir, epoch)  # 使用负准确率作为损失值
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        # 在每个epoch结束时更新学习率
        scheduler.step()
        if epoch == stop_epoch - 1:
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
    return model


class Tee:  # 新增Tee类用于双重输出
    def __init__(self, file_path, mode):
        self.file = open(file_path, mode)
        self.stdout = sys.stdout

    def write(self, text):
        self.file.write(text)
        self.stdout.write(text)
        self.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __del__(self):
        self.file.close()
        sys.stdout = self.stdout


# --- main function ---
if __name__ == '__main__':

    # fix seed
    seed = 1998
    # seed = random.randint(0, 10000)
    print("set seed = %d" % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # parser argument
    params = parse_args('train')
    print('--- Training ---\n')
    print(params)

    # 设置日志输出路径
    log_dir = os.path.join(params.save_dir, 'training_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_log_{time.strftime("%Y%m%d_%H%M%S")}.txt')

    # 重定向标准输出
    sys.stdout = Tee(log_file, 'w')

    # output dir
    params.tf_dir = '%s/log/%s' % (params.save_dir, params.name)
    params.checkpoint_dir = '%s/checkpoints/%s' % (params.save_dir, params.name)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # dataloader
    print('\n--- Prepare dataloader ---')
    print('\tbase with single seen domain {}'.format(params.dataset))
    print('\tval with single seen domain {}'.format(params.testset))
    base_file = os.path.join(params.data_dir, params.dataset, 'base.json')
    val_file = os.path.join(params.data_dir, params.testset, 'val.json')

    # model
    image_size = 224
    n_query = max(1, int(16 * params.test_n_way / params.train_n_way))
    base_datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.train_n_way, n_support=params.n_shot)
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
    val_datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.test_n_way, n_support=params.n_shot)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    model = MainMethod(model_dict[params.model], tf_path=params.tf_dir, n_way=params.train_n_way,
                       n_support=params.n_shot)
    model = model.cuda()

    # load model
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.resume != '':
        resume_file = get_resume_file('%s/checkpoints/%s' % (params.save_dir, params.resume), params.resume_epoch)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state'])
            print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
    else:
        if params.warmup == 'gg3b0':
            raise Exception('Must provide the pre-trained feature encoder file using --warmup option!')
        state = load_warmup_state('%s/checkpoints/%s' % (params.save_dir, params.warmup))
        model.feature.load_state_dict(state, strict=False)

    start = time.perf_counter()
    # training
    print('\n--- start the training ---')
    model = train(base_loader, val_loader, model, start_epoch, stop_epoch, params)
    end = time.perf_counter()
    print('Running time: %s Seconds: %s Min: %s Min per epoch' % (
        end - start, (end - start) / 60, (end - start) / 60 / params.stop_epoch))
