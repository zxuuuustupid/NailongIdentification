import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import models

from draw import process_show
from nailong.model.CNN import VGG
from nailong.model.dataset import SimpleImageFolderDataset

root_dir = r'D:\Projects\torch\nailong\dataset\train'
root_dir2 = r'D:\Projects\torch\nailong\dataset\test'
model_name = 'VGG11-Nailong'
epoch_num = 10
# 定义变换操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.RandomCrop(32, padding=4)
])
if __name__ == '__main__':
    # 创建Dataset实例
    dataset = SimpleImageFolderDataset(root_dir, transform=transform)
    dataset2 = SimpleImageFolderDataset(root_dir2, transform=transform)

    # 创建DataLoader实例
    data_loader1 = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    data_loader2 = DataLoader(dataset2, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device("cuda:0")
    print("load the model...")
    # model = CNN().to(device)
    # model = ResNet18().to(device)
    model = VGG().to(device)
    # model=models.resnet18().to(device)
    # criterion = TripletLoss()

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=5e-4)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=5e-4)
    loss_list = []
    train_acclist = []
    test_acclist = []
    # 现在你可以在训练循环中使用data_loader了
    for epoch in range(1, epoch_num + 1):
        train_loss = 0
        train_acc = 0
        model.train()
        # only_train_fc = True
        # if only_train_fc:
        #     for param in model.parameters():
        #         param.requires_grad_(False)
        # fc_in_features = model.fc.in_features
        # model.fc = torch.nn.Linear(fc_in_features, 10, bias=True).to(device)

        # if epoch == 30 or epoch == 50 or epoch==10 or epoch==70:
        #     optimizer.param_groups[0]['lr'] *= 0.1

        for images, label in data_loader1:
            img = images.to(device)
            label = label.to(device)
            out = model(img)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc

        test_acc = 0
        model.eval()
        for images2, label2 in data_loader2:
            img = images2.to(device)
            label = label2.to(device)
            out = model(img)
            loss = criterion(out, label)

            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            test_acc += acc

        train_acclist.append(train_acc / len(data_loader1))
        test_acclist.append(test_acc / len(data_loader2))
        loss_list.append(train_loss)

        torch.save(model.state_dict(), "model_weights.pth")
        print("Epoch:", epoch, "loss:{:.4f}  train acc:{:.4f}  test acc:{:.4f}".format(train_loss
                                                                                       , train_acc / len(data_loader1),
                                                                                       test_acc / len(data_loader2)))

    process_show(model_name, list(range(1, epoch_num + 1)), loss_list, train_acclist, test_acclist)



