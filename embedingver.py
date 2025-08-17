import torch 
import torch.nn as nn
from einops.layers.torch import Rearrange
import math

class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        """
        Args:
            dim (int): 埋め込みベクトルの次元数
        """
        super().__init__()
        self.dim = dim
        assert(self.dim % 2  == 0)

    def forward(self, time):
        """
        https://www.nomuyu.com/positional-encoding/
        Args:
            time (torch.Tensor): タイムステップのバッチ。shapeは (Batch, )
        
        Returns:
            torch.Tensor: 時間埋め込みベクトル。shapeは (Batch, dim)
        """
        # 現在の計算デバイス（CPU or GPU）を取得
        device = time.device
        
        # 埋め込み次元の半分

        half_dim = self.dim // 2
        
        # 埋め込みの周波数を計算 (logスケールで)
        # 数式: 1 / (10000^(2i / dim))
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # タイムステップと周波数を掛け合わせる
        # time: (B,) -> (B, 1)
        # embeddings: (D/2,) -> (1, D/2)
        # 結果: (B, D/2) (ブロードキャストを利用)
        embeddings = time.unsqueeze(1) * embeddings.unsqueeze(0)
        
        # sinとcosを計算し、連結する
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings


class ResNet(torch.nn.Module):
    def __init__(self, channels, timeembedding_dim):
        """ 
        timeembedding_dim: 埋め込みベクトルの次元
        """
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm = torch.nn.BatchNorm2d(channels)

        self.mlp = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(timeembedding_dim, channels)
        )
    def forward(self, x, timeembedding):
        assert(type(timeembedding) == torch.Tensor)
        "timeembeddingはすでにSinuを通ったもの"
        h = x
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + h
        x = self.batch_norm(x)
        x = self.relu(x)
        #print(f"resnet inside x = {x.shape}")
        x =x + self.mlp(timeembedding).unsqueeze(-1).unsqueeze(-1) #channelはそれ以外のサイズは変わらず
        #print(f"mlp shape = {self.mlp(timeembedding).unsqueeze(-1).unsqueeze(-1).shape}")
        return x

class NormalResnet(torch.nn.Module):
    def __init__(self, outdim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=outdim, out_channels=outdim, kernel_size=3, stride=1, padding=1)
        self.batch_norm = torch.nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h = x
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = x + h
        return x
def default(x, y):
    if x == None:
        return y
    return x

class Upsample(nn.Module):
    def __init__(self, dim, out_dim=None):
        super().__init__()
        self.fn = nn.Upsample(scale_factor=2, mode = 'nearest')
        self.fn2 = nn.Conv2d(in_channels=dim, out_channels=default(out_dim, dim), kernel_size=3, padding=1)

    def forward(self, x):
        x = self.fn(x)
        x = self.fn2(x)
        return x

class Downsample(nn.Module):
    def __init__(self, dim, out_dim=None):
        super().__init__()
        # rearrでチャネル数を4倍にした後, Convで元のチャネル数に戻す
        self.rearr = Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2)
        self.conv = nn.Conv2d(in_channels=4*dim,out_channels=default(out_dim, dim), kernel_size=3,padding=1)
    
    def forward(self, x):
        x = self.rearr(x)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, dim, model_channel=64, timeembedding_dim=128):
        super().__init__()
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv_0 = nn.Conv2d(dim, model_channel, 3, 1, padding=1)
        dim = model_channel
        self.resnet1_1 = ResNet(dim, timeembedding_dim=timeembedding_dim)
        self.resnet1_2 = ResNet(dim, timeembedding_dim=timeembedding_dim)
        self.downsample1 = Downsample(dim, dim*2)
        dim *= 2
        self.resnet2_1 = ResNet(dim, timeembedding_dim=timeembedding_dim)
        self.resnet2_2 = ResNet(dim, timeembedding_dim=timeembedding_dim)
        self.downsample2 = Downsample(dim, dim * 2)
        dim *= 2
        self.resnet3_1 = ResNet(dim, timeembedding_dim=timeembedding_dim)
        self.resnet3_2 = ResNet(dim, timeembedding_dim=timeembedding_dim)
        self.downsample3 = Downsample(dim, dim * 2)
        dim *=2
        self.resnet4_1 = ResNet(dim, timeembedding_dim=timeembedding_dim)
        self.resnet4_2 = ResNet(dim, timeembedding_dim=timeembedding_dim)
        self.downsample4 = Downsample(dim, dim * 2)
        dim *=2
        #print(f"down = {dim}") # 1024
        self.resnet5_1 = ResNet(dim, timeembedding_dim=timeembedding_dim)
        self.resnet5_2 = ResNet(dim, timeembedding_dim=timeembedding_dim)
        self.upsample1 = Upsample(dim, dim//2)
        dim //= 2
        # dim = 512
        self.resnet6_1 = ResNet(dim*2, timeembedding_dim=timeembedding_dim)
        self.resnet6_2 = ResNet(dim*2, timeembedding_dim=timeembedding_dim)
        self.upsample2 = Upsample(dim*2, dim//2)
        dim //= 2
        # dim = 256
        self.resnet7_1 = ResNet(dim*2, timeembedding_dim=timeembedding_dim)
        self.resnet7_2 = ResNet(dim*2, timeembedding_dim=timeembedding_dim)
        self.upsample3 = Upsample(dim*2, dim//2)
        dim //= 2
        #dim = 128
        self.resnet8_1 = ResNet(dim*2, timeembedding_dim=timeembedding_dim)
        self.resnet8_2 = ResNet(dim*2, timeembedding_dim=timeembedding_dim)
        self.upsample4 = Upsample(dim*2, dim // 2)
        dim //= 2
        self.resnet9_1 = ResNet(dim*2, timeembedding_dim=timeembedding_dim)
        self.resnet9_2 = ResNet(dim*2, timeembedding_dim=timeembedding_dim)
        #print(f" dim = {dim}")
        assert(model_channel == dim)
        self.conv_last = nn.Conv2d(dim*2, 3, 3, 1, padding=1)
        self.sinu = SinusoidalPositionEmbeddings(dim=timeembedding_dim)
        


    def forward(self, x, timestep):
        # timestep: torch.Tensor
        tmb = self.sinu(timestep)
        #print(f"tmb.shape = {tmb.shape}")
        x = self.conv_0(x)
        x = self.resnet1_1(x, tmb)
        x = self.resnet1_2(x, tmb)
        x1 = x
        #print(f"x1.shape = {x1.shape}")
        x = self.downsample1(x)
        x = self.resnet2_1(x, tmb)
        x = self.resnet2_2(x, tmb)
        x2 = x
        #print(f"x2.shape = {x2.shape}")
        x = self.downsample2(x)
        x = self.resnet3_1(x, tmb)
        x = self.resnet3_2(x, tmb)
        x3 = x
        x = self.downsample3(x)
        x = self.resnet4_1(x, tmb)
        x = self.resnet4_2(x, tmb)
        x4 = x
        x = self.downsample4(x)
        x = self.resnet5_1(x, tmb)
        x = self.upsample1(x)
        x = torch.cat((x4, x), dim = 1)
        x = self.resnet6_1(x, tmb)
        x = self.resnet6_2(x, tmb)  # <--- この行を追加
        x = self.upsample2(x)
        x = torch.cat((x3, x), dim = 1)
        x = self.resnet7_1(x, tmb)
        x = self.resnet7_2(x, tmb)  # <--- この行を追加
        x = self.upsample3(x)
        x = torch.cat((x2, x), dim = 1)
        x = self.resnet8_1(x, tmb)
        x = self.resnet8_2(x, tmb)  # <--- この行を追加
        x = self.upsample4(x)
        x = torch.cat((x1, x), dim = 1)
        x = self.resnet9_1(x, tmb)
        x = self.resnet9_2(x, tmb)  # <--- この行を追加
        x = self.conv_last(x)
        return x


# Kerasモデルの構造に合わせて書き直したSmallUNet
class SmallUNet(nn.Module):
    """
    Kerasで定義されたbuild_unetの構造をPyTorchで再実装したU-Netモデル。
    """
    def __init__(self, img_dim: int, timeembedding_dim: int = 128):
        """
        Args:
            img_dim (int): 入力画像のチャンネル数 (例: グレースケールなら1, カラーなら3)
            timeembedding_dim (int): タイムステップ埋め込みベクトルの次元数
        """
        super().__init__()
        
        # --- 1. タイムステップ埋め込み関連の層 ---
        self.sinu = SinusoidalPositionEmbeddings(dim=timeembedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(timeembedding_dim, timeembedding_dim),
            nn.GELU()
        )
        
        # --- 2. ダウンサンプリングパス (Encoder) ---
        # Level 1: (C, H, W) -> (64, H/2, W/2)
        self.down1_conv = nn.Sequential(
            # Kerasモデルに合わせて、時間埋め込みを結合したものを入力とする
            nn.Conv2d(in_channels=img_dim + timeembedding_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = Downsample(dim=64)

        # Level 2: (64, H/2, W/2) -> (128, H/4, W/4)
        self.down2_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = Downsample(dim=64)

        # --- 3. ボトルネック ---
        # (128, H/4, W/4) -> (256, H/4, W/4)
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # --- 4. アップサンプリングパス (Decoder) ---
        # Level 1: (256, H/4, W/4) -> (128, H/2, W/2)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1_conv = nn.Sequential(
            # スキップ接続でチャンネル数が (bottleneck_out + down2_out) = 256 + 128 = 384 になる
            nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Level 2: (128, H/2, W/2) -> (64, H, W)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2_conv = nn.Sequential(
            # スキップ接続でチャンネル数が (up1_out + down1_out) = 128 + 64 = 192 になる
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # --- 5. 最終層 ---
        # Kerasモデルに合わせて、出力は1チャンネルのノイズ画像
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): ノイズ入り画像 (Batch, Channels, Height, Width)
            timestep (torch.Tensor): タイムステップ (Batch, )
        
        Returns:
            torch.Tensor: 予測されたノイズ (Batch, 1, Height, Width)
        """
        # --- タイムステップ埋め込みの処理 ---
        tmb = self.sinu(timestep)
        tmb = self.time_mlp(tmb)
        
        # 埋め込みを画像の高さと幅に合わせる (B, D) -> (B, D, 1, 1) -> (B, D, H, W)
        B, C, H, W = x.shape
        tmb = tmb.view(B, -1, 1, 1).expand(B, -1, H, W)
        
        # 入力画像と時間埋め込みをチャンネル次元で結合
        x_with_time = torch.cat((x, tmb), dim=1)
        
        # --- ダウンサンプリング ---
        c1 = self.down1_conv(x_with_time) # スキップ接続用に保存
        p1 = self.pool1(c1)
        c2 = self.down2_conv(p1)          # スキップ接続用に保存
        p2 = self.pool2(c2)
        
        # --- ボトルネック ---
        bottom = self.bottleneck_conv(p2)
        
        # --- アップサンプリング ---
        u1 = self.up1(bottom)
        u1 = torch.cat((u1, c2), dim=1) # スキップ接続
        u1 = self.up1_conv(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat((u2, c1), dim=1) # スキップ接続
        u2 = self.up2_conv(u2)
        
        # --- 出力 ---
        output = self.out_conv(u2)
        
        return output

        
# class SmallUNet(nn.Module):
#     def __init__(self, dim, model_channel=64, timeembedding_dim=128):
#         super().__init__()
        
#         # --- 1. 初期畳み込み ---
#         self.conv_0 = nn.Conv2d(dim, out_channels=model_channel, kernel_size=1, stride=1, padding=0) #kernelsizeが1なのでpaddingは0
#         dim = model_channel

#         # --- 2. ダウンサンプリングパス (2段階) ---
#         # Level 1: 32x32 -> 16x16
#         self.resnet1_1 = ResNet(dim, timeembedding_dim=timeembedding_dim)
#         #self.resnet1_2 = ResNet(dim, timeembedding_dim=timeembedding_dim)
#         self.downsample1 = Downsample(dim, dim*2) #チャネル数を2倍
#         dim *= 2 # dim is now 128

#         # Level 2: 16x16 -> 8x8
#         self.resnet2_1 = NormalResnet(dim)
#         self.resnet2_2 = ResNet(dim, timeembedding_dim=timeembedding_dim)
#         self.downsample2 = Downsample(dim, dim * 2)
#         dim *= 2 # dim is now 256

#         # --- 3. ボトルネック ---
#         self.resnet_bottleneck1 = NormalResnet(dim)
#         self.resnet_bottleneck2 = ResNet(dim, timeembedding_dim=timeembedding_dim)

#         # --- 4. アップサンプリングパス (2段階) ---
#         # Level 2: 8x8 -> 16x16
#         self.upsample1 = Upsample(dim, dim//2)
#         dim //= 2 # dim is now 128
#         self.resnet3_1 = NormalResnet(dim*2)
#         self.resnet3_2 = ResNet(dim*2, timeembedding_dim=timeembedding_dim)

#         # Level 1: 16x16 -> 32x32
#         self.upsample2 = Upsample(dim*2, dim//2)
#         dim //= 2 # dim is now 64
#         self.resnet4_1 = NormalResnet(dim*2)
#         self.resnet4_2 = ResNet(dim*2, timeembedding_dim=timeembedding_dim)

#         # --- 5. 最終層 ---
#         assert(model_channel == dim)
        
#         self.conv_last = nn.Conv2d(dim*2, out_channels=1, kernel_size=1, stride=1, padding=0)
#         self.sinu = SinusoidalPositionEmbeddings(dim=timeembedding_dim)
        

#     def forward(self, x, timestep):
#         tmb = self.sinu(timestep)
        
#         # 初期畳み込み
#         x = self.conv_0(x)

#         # ダウンサンプリング
#         x = self.resnet1_1(x, tmb)
#         #x = self.resnet1_2(x, tmb)
#         x1 = x # 32x32
        
#         x = self.downsample1(x)
#         x = self.resnet2_1(x)
#         #x = self.resnet2_2(x, tmb)
#         x2 = x # 16x16

#         x = self.downsample2(x) # -> 8x8

#         # ボトルネック
#         x = self.resnet_bottleneck1(x)
#         #x = self.resnet_bottleneck2(x, tmb)

#         # アップサンプリング
#         x = self.upsample1(x) # -> 16x16
#         x = torch.cat((x2, x), dim = 1)
#         x = self.resnet3_1(x)
#         #x = self.resnet3_2(x, tmb)

#         x = self.upsample2(x) # -> 32x32
#         x = torch.cat((x1, x), dim = 1)
#         x = self.resnet4_1(x)
#         #x = self.resnet4_2(x, tmb)
        
#         # 最終出力
#         x = self.conv_last(x)
#         return x

        






