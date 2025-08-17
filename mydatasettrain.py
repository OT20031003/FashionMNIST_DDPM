from torch.nn import Module
import torch, math
from PIL import Image
from torchvision import transforms, datasets
import torchvision
from torch.utils.data import DataLoader
import random
from embedingver import ResNet, Downsample, Upsample, UNet, SinusoidalPositionEmbeddings, SmallUNet
import os
from torchvision.datasets import ImageFolder # ImageFolderをインポート
import SmallDDPM
import argparse


def FASHIONtraining(args, model, optimizer):
    model.train()
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),  # <-- この行を追加
        transforms.ToTensor(), 
        #transforms.Normalize((0.5,), (0.5, ))
    ])
    train_dataset = datasets.FashionMNIST(root='./FashionMNISTdata', train=True, download=True, transform=transform)
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"train dataset size = {len(train_dataset)}")
    num_epoch = 100
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epoch):
        for i, (images, _) in enumerate(train_loader):
            # images の shape: torch.Size([batch_size, 1, 28, 28])
            images = images*2 - 1
            optimizer.zero_grad()
            maxx = model.timesteps
            minn = 0
            t =  torch.randint(minn, maxx, (batch_size,), dtype=torch.long)

            
            epsilon = torch.randn_like(images)
            #print(f"images.shape = {images.shape}, t.shape = {t.shape}")
            predict = model.unet.forward(model.forward_process(images, t, noise=epsilon), t)
            loss = criterion(epsilon, predict)
            if (i + 1) % 50 == 0:
                
                print(f"Epoch [{epoch + 1}/{num_epoch}], Step [{i+1}/ {len(train_loader)}]")
                print(f"loss = {loss}")
                print(f"img min = {images.min()}, img max = {images.max()}")
                # if i > 300:
                #     break
                
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), 'smallmodel_weight.pth')


                
def Training(args, model, optimizer):
    model.train()
    dir_path = "./dataset" # dataset内のフォルダの画像
    epoch = 100
    sepoch = epoch
    files_file = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    while epoch >= 0:
        print(f"epoch = {sepoch - epoch}")
        for x in files_file:
            optimizer.zero_grad()
            print(x)
            x_0 = SmallDDPM.load_image_as_tensor(dir_path + "/" +x, clipdim=args.img_size)
            x_0 = x_0.unsqueeze(0)
            x_0 = x_0 * 2 - 1
            maxx = model.timesteps
            minn = 0
            t = torch.empty(1).uniform_(minn, maxx).long()
            print(t)
            epsilon = torch.randn(x_0.shape)
            
            predict = model.unet.forward(model.forward_process(x_0, t, noise=epsilon), t)
            criterion = torch.nn.MSELoss()
            loss = criterion(epsilon , predict)
            print(f"loss = {loss}")
            loss.backward()
            optimizer.step()
        epoch -= 1
    torch.save(model.state_dict(), 'prac_model_weight.pth')







def Training_test(args):
    model = SmallDDPM.GaussianDiffusion(channel_size=args.channel_num)
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-5)
    FASHIONtraining(args, model, optimizer)


def main():
    print("code execute!!")
    parser = argparse.ArgumentParser(description='Diffusion Training')
    parser.add_argument('--channel_num', type=int, help="dataset channel mnist->1", default=1)
    parser.add_argument('--img_size', type=int, help='dataset size', default=32)
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    args = parser.parse_args()
    Training_test(args)

if __name__ == '__main__':
    main()
# python mydatasettrain.py  --channel_num 1 --img_size 32 --batch_size 16  