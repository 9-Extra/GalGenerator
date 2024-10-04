import math
import os
from functools import partial
from typing import Optional

import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import DataLoader, Dataset

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

from common import DataSet, utils

# helpers functions

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return (t,) * length


def divisible_by(numer, denom):
    return (numer % denom) == 0


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# small helper modules

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim: int, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



# building block modules

class Block(Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            num_mem_kv=4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)
    
    

class Attend(nn.Module):
    def __init__(
            self,
            dropout=0.,
            flash=False,
            scale=None
    ):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)


    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        scale = default(self.scale, q.shape[-1] ** -0.5)

        # similarity

        sim = torch.einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = torch.einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out


class Attention(Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            num_mem_kv=4,
            flash=False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


# model

class Unet(Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            learned_variance=False,
            sinusoidal_pos_emb_theta=10000,
            attn_dim_head=32,
            attn_heads=4,
            full_attn=None,  # defaults to full attention only for inner most layer
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
                zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = Attention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                attn_klass(dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Attention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
                zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = Attention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                attn_klass(dim_out, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time):
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


# gaussian diffusion trainer class

def extract(a, t, x_shape) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
  

class GaussianDiffusion(Module):
    betas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    
    sqrt_recip_alphas_cumprod: torch.Tensor # alphas_cumprod倒数的开方
    sqrt_recipm1_alphas_cumprod: torch.Tensor # alphas_cumprod倒数 - 1的开方 
    
    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(
            image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size

        betas = linear_beta_schedule(timesteps)
  
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # 去掉alphas_cumprod最后一个值再在开头补一个1
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.num_timesteps = timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # 加的噪声比例
        register_buffer('betas', betas)
        # (1 - beta)的累乘，即每个时刻原图像的残留率
        register_buffer('alphas_cumprod', alphas_cumprod)
        # alphas_cumprod后移一个数
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        使用x_t，当前时间和噪声计算x_{t-1}
        """
        sqrt_recip_alphas_cumprod = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        return sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * noise

    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor):
        """
        使用x_t，x_{t-1}和时间计算噪声，是predict_start_from_noise的逆向过程，用于修复对x_t进行clip后改变的噪声
        """
        sqrt_recip_alphas_cumprod = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        return (sqrt_recip_alphas_cumprod * x_t - x0) / sqrt_recipm1_alphas_cumprod

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """
        返回指定时刻
        x_start
        """
        coef1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = extract(self.posterior_mean_coef2, t, x_t.shape) 
        
        # 后验均值
        posterior_mean = coef1 * x_start + coef2 * x_t
        # 后验方差
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
    
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, clip_x_start=False, rederive_pred_noise=False):
        """
        模型推理，得到预测的噪声和前一时刻的图像
        """      
        pred_noise = self.model(x, t)
   
        x_start = self.predict_start_from_noise(x, t, pred_noise)
   
        if clip_x_start:
            x_start = torch.clamp_(x_start, min=-1, max=1)
            if rederive_pred_noise:
                # x_start进行clamp后就与噪声不一致了，所以可能需要重新计算噪声
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def p_mean_variance(self, x: torch.Tensor, t: torch.Tensor, clip_denoised=True):
        pred_noise, x_start = self.model_predictions(x, t)
      
        if clip_denoised:
            x_start.clamp_(-1, 1)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int):
        """
        对分布p进行采样，即生成图像
        """
        b, *_ = x.shape
        batched_times = torch.full((b,), t, device=self.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times,
                                                                          clip_denoised=True)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start


    @torch.inference_mode()
    def sample(self, batch_size=16, return_all_timesteps=False):
        """
        生成图像
        """
        (h, w), channels = self.image_size, self.channels

        img = torch.randn((batch_size, channels, h, w), device=self.device)
        imgs = [img]

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample(img, t)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = unnormalize_to_zero_to_one(ret)
        return ret


    def q_sample(self, x_start, t, noise=None):
        """
        对分布q进行采样，即对图像加噪声
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None) -> torch.Tensor:
        """
        从图像和时间直接计算到loss
        """
        noise = default(noise, lambda: torch.randn_like(x_start))  # 随机噪声
        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # 按照时间对图像加噪
        
        model_out = self.model(x, t)  # 计算模型输出
        loss = F.mse_loss(model_out, noise) # L2损失
        return loss

    def forward(self, img, *args, **kwargs):
        b, c, h, w = img.shape
        device, img_size, = img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        # 随机获得时间
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)


def _train(
        model: GaussianDiffusion,
        data: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        save_dir: str
):
    model.train()
    device = next(model.parameters()).device
    print("Start Training")
    for e in range(1, epoch + 1):
        running_loss = []
        for batch in tqdm(data):
            batch = batch.to(device, non_blocking=True) / 255

            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()

            running_loss.append(loss)  # 保存引用，不获取loss的值以降低传输延迟

        pass  # 一个epoch结束

        mean_loss = sum(l.item() for l in running_loss) / len(data)
        print(f"Finished epoch {e}/{epoch}, mean-loss: {mean_loss}")

        if e % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"guass_diffusion_{e}.pth"))

    torch.save(model.state_dict(), os.path.join(save_dir, "guass_diffusion.pth"))
    
    pass


def guass_model(image_size: int, weight: Optional[str] = None) -> GaussianDiffusion:
    model = GaussianDiffusion(
        Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8)
        ),
        image_size=image_size,
        timesteps=1000,  # number of steps
    )

    model: GaussianDiffusion = torch.compile(model, fullgraph=True, disable=False) # noqa
    
    if weight is not None:
        model.load_state_dict(torch.load(weight, weights_only=True))

    return model


def train(
        dataset: str,
        weight: Optional[str],
        save: str,
        epoch: int
):
    device = torch.device("cuda")

    batch_size = 16
    # 加载数据集
    data_set = DataSet.H5Dataset(dataset)
    image_shape = data_set[0].shape
    assert image_shape[0] == 3 and image_shape[1] == image_shape[2]
    image_size = image_shape[1]
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             drop_last=True)
    
    # 创建文件夹
    dir_model = os.path.join(save, "gauss_ddpm_weights")
    os.makedirs(dir_model, exist_ok=True)

    print(f"图像大小为{image_size}")

    model = guass_model(image_size, weight)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    # 训练
    _train(model, data_loader, optimizer, epoch, dir_model)
    
    model = model.to(device).eval()
    count = 16
    images = (model.sample(count) * 255).permute((0, 2, 3, 1)).numpy(force=True)
    save = utils.auto_increase_dir(os.path.join(save, "result/guass"))
    for i in range(count):
        cv2.imwrite(os.path.join(save, f"gen{i}.png"), images[i, ...])

    pass

def sample(image_size: int, weight: str, count: int=16):
    device = torch.device("cuda")
    model = guass_model(image_size, weight).to(device).eval()
    
    images = (model.sample(count) * 255).permute((0, 2, 3, 1)).numpy(force=True)
    save = utils.auto_increase_dir("run/result/guass")
    for i in range(count):
        cv2.imwrite(os.path.join(save, f"gen{i}.png"), images[i, ...])

    

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    
    train("./run/datasets/anime_faces64.h5", None, save="./run", epoch=5)
    sample(64, "./run/gauss_ddpm_weights/guass_diffusion.pth")
