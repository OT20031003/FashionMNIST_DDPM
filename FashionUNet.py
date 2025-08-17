import torch.nn as nn
import torch
# from embedingver import SinusoidalPositionEmbeddings 
# 上記は未提供のため、仮のクラスを定義します
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(Block, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out

class FashionUNet(nn.Module):
    def __init__(self, inp_channel=1, time_embedding_dim=100, unet_dim=10, img_size=32):
        super().__init__()

        # --- 次元とサイズの設定 ---
        self.img_size = img_size
        
        # --- 時間エンベディング ---
        self.si = SinusoidalPositionEmbeddings(time_embedding_dim)
        
        # --- エンコーダー（ダウンサンプリング） ---
        # Level 1: 32x32
        self.te1 = self._make_te(time_embedding_dim, inp_channel)
        self.b1 = nn.Sequential(
            Block((inp_channel, self.img_size, self.img_size), inp_channel, unet_dim),
            Block((unet_dim, self.img_size, self.img_size), unet_dim, unet_dim),
            Block((unet_dim, self.img_size, self.img_size), unet_dim, unet_dim)
        )
        self.down1 = nn.Conv2d(unet_dim, unet_dim, 4, 2, 1)

        # Level 2: 16x16
        self.te2 = self._make_te(time_embedding_dim, unet_dim)
        self.b2 = nn.Sequential(
            Block((unet_dim, self.img_size // 2, self.img_size // 2), unet_dim, unet_dim * 2),
            Block((unet_dim * 2, self.img_size // 2, self.img_size // 2), unet_dim * 2, unet_dim * 2)
        )
        self.down2 = nn.Conv2d(unet_dim * 2, unet_dim * 2, 4, 2, 1)

        # Level 3: 8x8
        self.te3 = self._make_te(time_embedding_dim, unet_dim * 2)
        self.b3 = nn.Sequential(
            Block((unet_dim * 2, self.img_size // 4, self.img_size // 4), unet_dim * 2, unet_dim * 4),
            Block((unet_dim * 4, self.img_size // 4, self.img_size // 4), unet_dim * 4, unet_dim * 4)
        )
        self.down3 = nn.Conv2d(unet_dim * 4, unet_dim * 4, 4, 2, 1)

        # --- ボトルネック ---
        # Level 4: 4x4
        self.te_mid = self._make_te(time_embedding_dim, unet_dim * 4)
        self.b_mid = nn.Sequential(
            Block((unet_dim * 4, self.img_size // 8, self.img_size // 8), unet_dim * 4, unet_dim * 4),
            Block((unet_dim * 4, self.img_size // 8, self.img_size // 8), unet_dim * 4, unet_dim * 4)
        )

        # --- デコーダー（アップサンプリング） ---
        # Level 3: 8x8
        self.up1 = nn.ConvTranspose2d(unet_dim * 4, unet_dim * 4, 4, 2, 1)
        self.te4 = self._make_te(time_embedding_dim, unet_dim * 8) # skip connectionでチャンネル数が倍になる
        self.b4 = nn.Sequential(
            Block((unet_dim * 8, self.img_size // 4, self.img_size // 4), unet_dim * 8, unet_dim * 4),
            Block((unet_dim * 4, self.img_size // 4, self.img_size // 4), unet_dim * 4, unet_dim * 2)
        )

        # Level 2: 16x16
        self.up2 = nn.ConvTranspose2d(unet_dim * 2, unet_dim * 2, 4, 2, 1)
        self.te5 = self._make_te(time_embedding_dim, unet_dim * 4)
        self.b5 = nn.Sequential(
            Block((unet_dim * 4, self.img_size // 2, self.img_size // 2), unet_dim * 4, unet_dim * 2),
            Block((unet_dim * 2, self.img_size // 2, self.img_size // 2), unet_dim * 2, unet_dim)
        )

        # Level 1: 32x32
        self.up3 = nn.ConvTranspose2d(unet_dim, unet_dim, 4, 2, 1)
        self.te_out = self._make_te(time_embedding_dim, unet_dim * 2)
        self.b_out = nn.Sequential(
            Block((unet_dim * 2, self.img_size, self.img_size), unet_dim * 2, unet_dim),
            Block((unet_dim, self.img_size, self.img_size), unet_dim, unet_dim),
            Block((unet_dim, self.img_size, self.img_size), unet_dim, unet_dim, normalize=False)
        )

        # --- 出力層 ---
        self.conv_out = nn.Conv2d(unet_dim, inp_channel, 3, 1, 1)

    def forward(self, x, timestep):
        n = len(x)
        
        # 1. 時間エンベディングの計算
        t = self.si(timestep) # [batch, time_embedding_dim]
        #print(f"t.shape = {t.shape}")
        # 2. エンコーダー
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1)) # スキップ接続用に保持
        #print(f"out1.shape = {out1.shape}")
        down1_out = self.down1(out1)
        out2 = self.b2(down1_out + self.te2(t).reshape(n, -1, 1, 1)) # スキップ接続用に保持

        down2_out = self.down2(out2)
        out3 = self.b3(down2_out + self.te3(t).reshape(n, -1, 1, 1)) # スキップ接続用に保持

        down3_out = self.down3(out3)
        
        # 3. ボトルネック
        out_mid = self.b_mid(down3_out + self.te_mid(t).reshape(n, -1, 1, 1))
        
        # 4. デコーダー
        up1_out = self.up1(out_mid)
        # スキップ接続: out3と結合
        cat1 = torch.cat((up1_out, out3), dim=1) 
        out4 = self.b4(cat1 + self.te4(t).reshape(n, -1, 1, 1))
        
        up2_out = self.up2(out4)
        # スキップ接続: out2と結合
        cat2 = torch.cat((up2_out, out2), dim=1)
        out5 = self.b5(cat2 + self.te5(t).reshape(n, -1, 1, 1))

        up3_out = self.up3(out5)
        # スキップ接続: out1と結合
        cat3 = torch.cat((up3_out, out1), dim=1)
        out_final = self.b_out(cat3 + self.te_out(t).reshape(n, -1, 1, 1))
        
        # 5. 出力
        out = self.conv_out(out_final)
        return out

    def _make_te(self, dim_in, dim_out):
        # 時間エンベディングを各層のチャンネル数に合わせるための小さなネットワーク
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

# --- モデルの動作確認 ---
if __name__ == '__main__':
    # パラメータ設定
    batch_size = 8
    img_size = 32
    inp_channel = 1
    
    # モデルのインスタンス化
    model = FashionUNet(inp_channel=inp_channel, img_size=img_size)
    
    # ダミーデータの作成
    x = torch.randn(batch_size, inp_channel, img_size, img_size)
    timestep = torch.randint(1, 1000, (batch_size,))
    print(f"timestep = {timestep}")
    # フォワードパスの実行
    output = model(x, timestep)
    
    # 出力サイズの確認
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    assert output.shape == x.shape, "Output shape does not match input shape!"
    print("Shape check passed!")