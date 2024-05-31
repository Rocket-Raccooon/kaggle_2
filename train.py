import torch
from data import CAPTCHALoader
from torch.utils.data import DataLoader
from torchvision import models as torch_models
from torch import nn
import argparse
import sys
import time
import os
import torch.optim as optim
from tqdm import tqdm


def parse_args(args):
    parser = argparse.ArgumentParser(description='Fashion MNIST predictor train')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch size for training')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--model_name', type=str, default='resnet34')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam', help='Type of the optimizer')

    return parser.parse_args(args)


# класс для нормализации изображения
# с его помощью можно нормализацию делать сразу на девайте
class Normalization(object):
    def __init__(self):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:0'
        elif torch.backends.mps.is_available():
            device = 'mps'
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device) * 255
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device) * 255


    def __call__(self, batch):
        batch = (batch - self.mean) / self.std
        batch = batch.permute(0, 3, 1, 2)  # NxHxWxC -> NxCxHxW
        return batch


# функция получения модели, поддерживает resnet, densenet, vgg, mobilenet_v2, mnasnet
def get_model(model_name, n_classes=2):

    net = getattr(torch_models, model_name)(pretrained=True)

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
    elif model_name.startswith('vit'):
        classifier = nn.Linear(net.heads.head.in_features, n_classes)
        net.heads.head = classifier
    else:
        raise AttributeError(f'{model_name} is unknown model name')

    net.eval()
    return net, classifier


def save_checkpoint(checkpoint):
    # save weights as fp32
    for key, val in checkpoint['model_weight'].items():
        checkpoint['model_weight'][key] = val.float().cpu()
    try:
        torch.save(checkpoint, checkpoint['fname'])
    except IOError:
        time.sleep(5)
        torch.save(checkpoint, checkpoint['fname'])


def train(args):
    # папка записи результата и файл с логами
    os.makedirs(f'weights_{args.model_name}', exist_ok=True)
    log_file = open(f'weights_{args.model_name}/logs.txt', 'w')

    # инициализация датасетов
    val_dataset = CAPTCHALoader(data_path='./images.npy', mode='val', labels_path='./labels.npy')
    train_dataset = CAPTCHALoader(data_path='./images.npy', mode='train', labels_path='./labels.npy')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset,  batch_size=args.batch_size, shuffle=False)

    # получаем модель
    net, classifier = get_model(args.model_name, train_dataset.n_classes)
    # инициализируем оптимизатор
    if args.optimizer == 'sgd':
        print('Using sgd optimizer')
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        print('Using adam optimizer')
        # optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # уменьшаем постепенно скорость обучения по косинусу
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    criterion = nn.CrossEntropyLoss()
    normalizator = Normalization()

    # определяем целевой девайс
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available():
        device = 'mps'
    net = net.to(device)

    best_val_loss, best_epoch, best_val_acc = 1e10, 0, 0
    for epoch in range(1, args.n_epochs + 1):

        # classifier.train()
        net.train()
        train_epoch_loss = val_epoch_loss = train_corr_preds = val_corr_preds = 0
        for images, targets in tqdm(train_loader, desc='train'):
            images, targets = images.to(device), targets.to(device)
            # forward pass
            images = normalizator(images)
            out = net(images)
            # подсчет лосса
            loss_iter = criterion(out, targets)

            # backward pass
            optimizer.zero_grad()
            loss_iter.backward()
            optimizer.step()

            # подсчитываем суммарный лосс за эпоху и кол-во верных предсказаний
            train_epoch_loss += loss_iter.item()
            preds = out.argmax(dim=1)
            train_corr_preds += torch.sum(preds == targets).item()

        train_epoch_loss /= len(train_loader.dataset)
        train_corr_preds /= len(train_loader.dataset)

        # classifier.eval()
        net.eval()
        for images, targets in tqdm(val_loader, desc='val'):
            images, targets = images.to(device), targets.to(device)

            images = normalizator(images)
            # получение предсказаний и подсчет лосса без расчета градиентов
            with torch.no_grad():
                out = net(images)
                loss_iter = criterion(out, targets)

            # подсчитываем суммарный лосс за эпоху и кол-во верных предсказаний
            val_epoch_loss += loss_iter.item()
            preds = out.argmax(dim=1)
            val_corr_preds += torch.sum(preds == targets).item()

        val_epoch_loss /= len(val_loader.dataset)
        val_corr_preds /= len(val_loader.dataset)

        if val_corr_preds * 100 > best_val_acc:
            best_val_loss = val_epoch_loss
            best_val_acc = val_corr_preds * 100
            best_epoch = epoch

            fname = f'weights_{args.model_name}/epoch_{epoch}_checkpoint.pth'
            model_weight = net.state_dict()
            checkpoint = {
                'model_weight': model_weight,
                'lr_scheduler': lr_scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'fname': fname,
                'train_loss': train_epoch_loss,
                'train_corr_preds': train_corr_preds,
                'val_loss': val_epoch_loss,
                'val_corr_preds': val_corr_preds,
                'arguments': args,
            }
            save_checkpoint(checkpoint)


        log_str = f'epoch {epoch} | train_loss = {train_epoch_loss} | train acc = {train_corr_preds * 100:0.1f} % | ' \
                  f'val_loss = {val_epoch_loss} | val acc = {val_corr_preds * 100:0.1f}%\n'
        print(log_str, end='')
        log_file.write(log_str)

    log_str = f'best epoch = {best_epoch} | best val loss = {best_val_loss:0.2f} | best_val_acc = {best_val_acc:0.1f} %'
    print(log_str)
    log_file.write(log_str)
    log_file.close()


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    train(args)
