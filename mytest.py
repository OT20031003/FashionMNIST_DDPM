from torch.nn import Module
import torch, math
from PIL import Image
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import random
from embedingver import ResNet, Downsample, Upsample, UNet, SinusoidalPositionEmbeddings, SmallUNet
import os
from torchvision.datasets import ImageFolder # ImageFolderをインポート
import SmallDDPM
import argparse







def InferTest(args):
    """
    学習済みモデルを使い、ノイズから画像を生成して保存する関数。
    """
    # 1. デバイスの設定（GPUが利用可能ならGPUを使用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. モデルのインスタンス化と学習済み重みのロード
    model = SmallDDPM.GaussianDiffusion(channel_size=args.channel_num)
    
    # Training関数で保存されるファイル名 'model_weight.pth' を指定
    # map_location=device を使うことで、GPUがない環境でもGPUで学習したモデルを読み込める
    try:
        model.load_state_dict(torch.load('./smallmodel_weight.pth', map_location=device))
    except FileNotFoundError:
        print("Error: 'model_weight.pth' not found.")
        print("Please train the model first by uncommenting and running the Training() function in main().")
        return
        
    model.to(device)
    model.eval() # モデルを評価モードに設定（Dropoutなどを無効化）

    # 3. 画像生成の準備
    # Training時の画像サイズ（CenterCrop(256)）に合わせる
    image_size = args.img_size
    channels = args.channel_num
    batch_size = args.batch_size # 一度に生成する画像の枚数

    # 4. 逆拡散プロセスによる画像生成
    print("Generating image from pure noise...")
    
    # (batch_size, channels, height, width) の形状でランダムノイズを生成
    img = torch.randn((batch_size, channels, image_size, image_size), device=device)
    # img =  load_image_as_tensor("./a.png").unsqueeze(0).to(device)
    # img = 2*img - 1
    SmallDDPM.save_tensor_as_image(img.squeeze(0), "rand.png")
    # 勾配計算は不要なため、torch.no_grad()コンテキストで実行
    with torch.no_grad():
        # model.timesteps - 1 から 0 までループ
        tss = torch.tensor([model.timesteps])
        img2 = model.reverse_process(img, tss)
        
    print("Image generation complete.")

    # 5. 生成した画像を保存
    # [-1, 1] の範囲で出力される画像を [0, 1] に変換し、PILで扱えるようにCPUに送る
    if torch.isnan(img2).any():
        print("NaN detected in generated image!")
    if torch.isinf(img2).any():
        print("Inf detected in generated image!")

    #generated_image = (img2.clamp(-1, 1) + 1) / 2
    
    SmallDDPM.save_tensor_as_image(img2.squeeze(0).cpu(), "generated_image.png")




def main():
    print("code execute!!")
    parser = argparse.ArgumentParser(description="DIffusion")
    parser.add_argument('--channel_num', type=int, help="dataset channel mnist->1", default=1)
    parser.add_argument('--img_size', type=int, help='dataset size', default=32)
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    args = parser.parse_args()
    InferTest(args)

if __name__ == '__main__':
    main()
# python mytest.py --channel_num 1 --img_size 32 --batch_size 1