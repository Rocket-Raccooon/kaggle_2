import torch
from torchvision import models as torch_models
from data import CAPTCHALoader
from torch import nn
import argparse
import sys
from tqdm import tqdm
from os import path as osp
import pandas as pd
from train import Normalization
import torch.nn.functional as F
from torch.utils.data import DataLoader


def parse_args(args):
    parser = argparse.ArgumentParser(description='Fashion MNIST predictor train')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch size for training')
    parser.add_argument('--tta', action='store_true', help='enable test time augmentation')
    parser.add_argument('--model_log', type=str, default='weights_resnet34/logs.txt', help='path to the log file')
    parser.add_argument('--n_classes', type=int, default=26)

    return parser.parse_args(args)


# загрузка обученной модели
def get_model(model_log, n_classes):
    # по пути до лога понимаем архитектуру модели
    model_name = osp.basename(osp.dirname(model_log)).replace('weights_', '')
    net = getattr(torch_models, model_name)(pretrained=False)

    # create classifier for different models
    if model_name.startswith('resnet') or model_name.startswith('shufflenet') or model_name.startswith('regnet'):
        classifier = nn.Linear(net.fc.in_features, n_classes)
        net.fc = classifier
    elif model_name.startswith('densenet'):
        classifier = nn.Linear(net.classifier.in_features, n_classes)
        net.classifier = classifier
    elif model_name.startswith('vgg'):
        net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        classifier = nn.Linear(512, n_classes)
        net.classifier = classifier
    elif model_name.startswith('mobilenet_v2') or model_name.startswith('mnasnet') or model_name.startswith('efficientnet'):
        classifier = nn.Linear(net.classifier[-1].in_features, n_classes)
        net.classifier = classifier
    else:
        raise AttributeError(f'{model_name} is unknown model name')

    # понимаем какая эпоха была лучшая и подгружаем именно ее
    log_last_line = open(model_log).readlines()[-1]
    best_epoch = int(log_last_line.split('best epoch = ')[1].split(' ')[0])
    fname_checkpoint = f'{osp.dirname(model_log)}/epoch_{best_epoch}_checkpoint.pth'
    print(f'load {fname_checkpoint}')
    checkpoint = torch.load(fname_checkpoint)
    weights = checkpoint['model_weight']

    print(net.load_state_dict(weights))
    net.eval()
    return net


def test(args):
    # загружаем сеть, датасет, нормализатор
    net = get_model(args.model_log, args.n_classes)
    test_dataset = CAPTCHALoader(data_path='./images_sub.npy', mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    normalizator = Normalization()

    # определяем целевой девайс
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available():
        device = 'mps'
    net = net.to(device)
    # структура данных нужная для отправки сабмита
    out_data = {
        'Id': [],
        'Category': []
    }
    out_scores = {
        'Id': [],
        'scores': []
    }
    test_id_counter = 0

    # classifier.eval()
    net.eval()
    for images in tqdm(test_loader, desc='test'):
        images = images.to(device)

        images = normalizator(images)
        with torch.no_grad():
            outs = F.log_softmax(net(images), dim=1)
            if args.tta:
                # делаем предсказание для отзеркаленных данных и складываем их
                outs += F.log_softmax(net(torch.fliplr(images)), dim=1)

        preds = outs.argmax(dim=1)

        for pred, out in zip(preds, outs):
            out_data['Id'].append(test_id_counter)
            out_data['Category'].append(pred.item())

            out_scores['Id'].append(test_id_counter)
            out_scores['scores'].append(out.cpu().tolist())

            test_id_counter += 1

    df = pd.DataFrame(out_data)
    df.to_csv(f'{osp.dirname(args.model_log)}_sub.csv', index=None)

    df = pd.DataFrame(out_scores)
    df.to_csv(f'{osp.dirname(args.model_log)}_scores.csv', index=None)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    test(args)
