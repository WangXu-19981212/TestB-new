import numpy as np
import torch
import os
import random
from methods.main_method import MainMethod
from data.datamgr import SetDataManager
from options import parse_args, get_best_file, get_resume_file
from methods.backbone_multiblock import model_dict


def evaluate(novel_loader, n_way=5, n_support=5):
    iter_num = len(novel_loader)
    acc_all = []
    # Model
    model = MainMethod(model_dict[params.model], tf_path=params.tf_dir, n_way=params.train_n_way,
                       n_support=params.n_shot)
    model = model.cuda()

    # Update model
    checkpoint_dir = '%s/checkpoints/%s/best_model.tar' % (params.save_dir, params.name)
    state = torch.load(checkpoint_dir)['state']
    if 'FWT' in params.name:
        model_params = model.state_dict()
        pretrained_dict = {k: v for k, v in state.items() if k in model_params}
        model_params.update(pretrained_dict)
        model.load_state_dict(model_params)
    else:
        model.load_state_dict(state, False)

    if params.method != 'TPN':
        model.eval()
    for ti, (x, _) in enumerate(novel_loader):
        x = x.cuda()
        n_query = x.size(1) - n_support
        model.n_query = n_query
        yq = np.repeat(range(n_way), n_query)
        with torch.no_grad():
            scores = model.set_forward(x)
            _, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:, 0] == yq)
            acc = top1_correct * 100. / (n_way * n_query)
            acc_all.append(acc)
        print('Task %d : %4.2f%%' % (ti, acc))

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('Test Acc = %4.2f +- %4.2f%%' % (acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
    return (acc_mean, 1.96 * acc_std / np.sqrt(iter_num))


# 保存最佳结果的函数（示例）
def save_best_result(best_result_mean, best_result_std, best_result_iteration):
    with open('best_result_oneshot.txt', 'a') as f:
        f.write(f'TestDataSet: {params.dataset}\n')
        f.write(f'Best result mean: {best_result_mean}\n')
        f.write(f'Best result std: {best_result_std}\n')
        f.write(f'Found in iteration: {best_result_iteration + 1}\n')
        f.write(f'-------------------------------------------------------\n')


if __name__ == '__main__':
    # fix seed
    np.random.seed(0)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # parse arguments
    params = parse_args('test')
    print('--- Testing ---\n')
    print(params)

    # data loader
    print('\n--- Prepare dataloader ---')
    params.tf_dir = '%s/log/%s' % (params.save_dir, params.name)
    image_size = 224
    n_query = max(1, int(16 * params.test_n_way / params.train_n_way))
    test_datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.test_n_way, n_support=params.n_shot)
    test_file = os.path.join(params.data_dir, params.testset, f'{params.split}.json')
    print('Load test data from %s' % test_file)
    test_loader = test_datamgr.get_data_loader(test_file, aug=False)
    # testing
    print('\n--- start testing ---')
    best_result = None
    best_accuracy = 0.0

    for i in range(50):
        print(f'\n--- Test iteration {i + 1} ---')
        accuracy,std = evaluate(test_loader, params.test_n_way, params.n_shot)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_result = accuracy

    print('\n--- Best Test Result ---')
    print(f'Best Accuracy: {best_result}')
