import torch
import torch.nn as nn
import torch.nn.functional as F


class Diffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape):
        b = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_t * noise

    def p_losses(self, model, x0: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0=x0, t=t, noise=noise)
        noise_pred = model(x_noisy, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if torch.all(t == 0):
            return model_mean

        posterior_var_t = self.extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
        return model_mean + nonzero_mask * torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, model, batch_size: int, image_size: int, channels: int = 3, device="cuda"):
        model.eval()
        x = torch.randn(batch_size, channels, image_size, image_size, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)

        return x
