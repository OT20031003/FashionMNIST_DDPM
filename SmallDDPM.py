from torch.nn import Module
import torch
from torchvision import transforms
from PIL import Image
from embedingver import ResNet, Downsample, Upsample, SinusoidalPositionEmbeddings, SmallUNet
from mnist_unet import MnistUNet
from FashionUNet import FashionUNet
def make_beta_schedule(timesteps, start_beta, end_beta):
    a = (end_beta - start_beta ) /timesteps
    #y = a * time + start_beta
    sch = []
    for t in range(timesteps):
        assert(a*t + start_beta < 1)
        sch.append(a * t + start_beta)
    return sch  

def load_image_as_tensor(image_path:str, clipdim=32)->torch.Tensor:
    try:
        pil_img = Image.open(image_path)
        # 256にクリップ
        transform_clip = transforms.CenterCrop(clipdim)
        transform = transforms.ToTensor()
        tensor_img = transform(pil_img)
        tensor_img = transform_clip(tensor_img)

        return tensor_img
    except FileNotFoundError:
        print(f"The file at {image_path} was not found")

def save_tensor_as_image(tensor: torch.Tensor, save_path: str):
    try:
        # DDPMの出力が[-1, 1]の場合、[0, 1]に正規化する
        # (x + 1) / 2 は [-1, 1]の値を [0, 1]に変換する一般的な方法です。
        tensor = (tensor - tensor.min())/(tensor.max() - tensor.min())
            
        # ToPILImage()は、(C, H, W)のテンソルをPIL画像オブジェクトに変換します。
        # 入力テンソルの値は、0.0-1.0の範囲である必要があります。
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(tensor)
        
        # PILのsaveメソッドで画像を保存
        pil_image.save(save_path)
        print(f"Image successfully saved to {save_path}")
        
    except Exception as e:
        print(f"An error occurred while saving the image: {e}")



class GaussianDiffusion(Module):
    def __init__(
            self, 
            timesteps = 1000, 
            start_beta = 0.0001,
            end_beta = 0.02, 
            channel_size = 3
    ):
        super().__init__()
        self.beta_schedules = make_beta_schedule(timesteps, start_beta, end_beta)
        self.alpha_schedules = []
        for i in range(timesteps):
            self.alpha_schedules.append(1 - self.beta_schedules[i])
        self.alpha_bar_schedules = []
        tmp = 1
        for i in range(timesteps):
            self.alpha_bar_schedules.append(tmp * self.alpha_schedules[i])
            tmp *= self.alpha_schedules[i]
        #print(alpha_bar_schedules)
        self.schedule = torch.tensor(self.alpha_bar_schedules)
        self.register_buffer('betas', torch.tensor(self.beta_schedules))
        self.register_buffer('alphas', torch.tensor(self.alpha_schedules))
        self.register_buffer('alpha_bars', torch.tensor(self.alpha_bar_schedules))
        self.unet = FashionUNet(inp_channel=channel_size) # ここは白黒かどうかで分ける
        #print(f"MNISTUNET = {MnistUNet}")
        self.timesteps = timesteps
        #print(type(self.schedule))
    
    def forward_process(self, img, timestep, noise = None):
        assert(type(img) == torch.Tensor)
        if noise == None:
            noise = torch.randn_like(img)
        assert(img.shape[0] == timestep.shape[0])
        noise = noise.clip(-1.0, 1.0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[timestep]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar  = torch.sqrt(1 - self.alpha_bars[timestep]).view(-1, 1, 1, 1)
        img2 = sqrt_alpha_bar * img + sqrt_one_minus_alpha_bar * noise
        
        return img2
    
    

    def reverse_onestep(self, img, timestep):
        """ 1ステップの逆拡散 """
        # timestep は torch.tensor([999]) のような形式で渡されることを想定
        
        # 予測されたノイズを取得
        epsilon_theta = self.unet.forward(img, timestep)
        
        # 各種パラメータを取得
        alpha_t = self.alphas[timestep]
        alpha_bar_t = self.alpha_bars[timestep]
        beta_t = self.betas[timestep]
        
        # ノイズを生成
        z = torch.randn_like(img)
        
        # 最後のステップ (t=0) ではノイズを加えない
        if timestep.item() == 0:
            z = torch.zeros_like(img)
        z = z.clip(-1.0, 1.0)    
        sigma_t = torch.sqrt(beta_t)
        
        
        term1 = 1 / torch.sqrt(alpha_t)
        term2 = (img - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * epsilon_theta)
        ans = term1 * term2 + sigma_t * z
        ans = ans.clip(-1.0, 1.0)
        return ans

    def reverse_process(self, img, timestep):
        """ timestepから0になるまで逆拡散を繰り返す """
        
        # 正しい型チェック (isinstanceを使用)
        save_tensor_as_image(img.squeeze(0), "./result/ongo_start.png")
        if not isinstance(timestep, torch.Tensor):
            print(f"エラー: timestepはTensorである必要がありますが、{type(timestep)}が渡されました。")
            raise TypeError("timestep must be a torch.Tensor")

        # .item() を使ってTensorからPythonの数値を取得
        ts = timestep.item()

        # ts から 0 までループ
        # Python 3のrangeでは逆順のループは range(start, stop, step) を使う
        for current_t in range(ts-1, -1, -1):
            # 現在のタイムステップをTensorに変換して渡す
            current_t_tensor = torch.tensor([current_t], device=img.device)
            img = self.reverse_onestep(img, current_t_tensor)
            if current_t %50==0:
                print(f"current_t = {current_t}")
                if torch.isnan(img).any():
                    print("NaN detected in generated image!")
                if torch.isinf(img).any():
                    print("Inf detected in generated image!")
                
                save_tensor_as_image(img.squeeze(0), "./result/ongo" + str(current_t)+".png")
            
        return img
