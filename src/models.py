from .runtime import *
class Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 784), nn.Sigmoid())
    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                                     nn.Linear(256, latent_dim), nn.ReLU())
        self.decoder = Decoder(latent_dim)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder = Decoder(latent_dim)
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)
    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        return self.decoder(z), mu, logvar

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
    def features(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x.flatten(1)))))
    def forward(self, x):
        return self.fc3(self.features(x))  # Logits, tanpa softmax untuk CrossEntropyLoss.

class VanillaGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
                                 nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
                                 nn.Linear(512, 784), nn.Tanh())
    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class VanillaDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.LeakyReLU(0.2),
                                 nn.Dropout(0.3), nn.Linear(512, 256), nn.LeakyReLU(0.2),
                                 nn.Linear(256, 1))
    def forward(self, x):
        return self.net(x).flatten()

class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 7, 1, 0, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())
    def forward(self, z):
        return self.net(z[:, :, None, None])  # 1 -> 7 -> 14 -> 28.

class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 7, 1, 0))
    def forward(self, x):
        return self.net(x).flatten()  # 28 -> 14 -> 7 -> 1, satu logit per citra.

def dcgan_init(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)

class TimeEmbedding(nn.Module):
    def __init__(self, size=32):
        super().__init__()
        self.size = size
    def forward(self, t):
        frequencies = torch.exp(-np.log(10000) * torch.arange(
            self.size // 2, device=t.device) / (self.size // 2 - 1))
        angles = t.float()[:, None] * frequencies[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=128):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t)[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))

class SmallUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.time = nn.Sequential(TimeEmbedding(), nn.Linear(32, 128), nn.SiLU(), nn.Linear(128, 128))
        self.input = nn.Conv2d(1, base, 3, padding=1)
        self.block1 = ResidualBlock(base, base)
        self.down1 = nn.Conv2d(base, base * 2, 4, 2, 1)
        self.block2 = ResidualBlock(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, 2, 1)
        self.middle = ResidualBlock(base * 4, base * 4)
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1)
        self.block3 = ResidualBlock(base * 4, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.block4 = ResidualBlock(base * 2, base)
        self.output = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 1, 3, padding=1))
    def forward(self, x, t):
        emb = self.time(t)
        a = self.block1(self.input(x), emb)
        b = self.block2(self.down1(a), emb)
        h = self.middle(self.down2(b), emb)
        h = self.block3(torch.cat([self.up1(h), b], 1), emb)
        h = self.block4(torch.cat([self.up2(h), a], 1), emb)
        return self.output(h)

class DDPM(nn.Module):
    def __init__(self, steps=1000, base=32):
        super().__init__()
        self.net = SmallUNet(base)
        self.steps = steps
        beta = torch.linspace(1e-4, 0.02, steps)
        alpha = 1 - beta
        abar = torch.cumprod(alpha, dim=0)
        previous = torch.cat([torch.ones(1), abar[:-1]])
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("abar", abar)
        self.register_buffer("posterior_var", beta * (1 - previous) / (1 - abar))
        self.register_buffer("coef_x0", beta * previous.sqrt() / (1 - abar))
        self.register_buffer("coef_xt", (1 - previous) * alpha.sqrt() / (1 - abar))
    def loss(self, x):
        x0 = x * 2 - 1
        t = torch.randint(self.steps, (len(x),), device=x.device)
        noise = torch.randn_like(x0)
        a = self.abar[t][:, None, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        return F.mse_loss(self.net(xt, t), noise)
    @torch.no_grad()
    def sample(self, n, initial_noise=None, trajectory=False):
        x = torch.randn(n, 1, 28, 28, device=self.beta.device) if initial_noise is None else initial_noise.clone()
        keep = {self.steps, int(self.steps * .8), int(self.steps * .6),
                int(self.steps * .4), int(self.steps * .2), 0}
        states = {self.steps: x[:1].cpu()} if trajectory else {}
        for i in reversed(range(self.steps)):
            t = torch.full((n,), i, device=x.device, dtype=torch.long)
            eps = self.net(x, t)
            x0_hat = ((x - (1 - self.abar[i]).sqrt() * eps) / self.abar[i].sqrt()).clamp(-1, 1)
            mean = self.coef_x0[i] * x0_hat + self.coef_xt[i] * x
            x = mean + self.posterior_var[i].sqrt() * torch.randn_like(x) if i > 0 else mean
            if trajectory and i in keep:
                states[i] = x[:1].cpu()
        result = ((x + 1) / 2).clamp(0, 1)
        return (result, states) if trajectory else result

def make_model(name):
    factories = {
        "ae": lambda: Autoencoder(CFG["latent_ae"]),
        "sparse": lambda: Autoencoder(CFG["latent_ae"]),
        "vae": lambda: VAE(CFG["latent_ae"]),
        "classifier": MLPClassifier, "frozen_decoder": lambda: Decoder(64),
        "ddpm": lambda: DDPM(CFG["diffusion_steps"], CFG["unet_channels"]),
    }
    if name in ["gan", "dcgan"]:
        g = VanillaGenerator() if name == "gan" else DCGenerator()
        d = VanillaDiscriminator() if name == "gan" else DCDiscriminator()
        if name == "dcgan":
            g.apply(dcgan_init)
            d.apply(dcgan_init)
        return g.to(DEVICE), d.to(DEVICE)
    return factories[name]().to(DEVICE)

def parameter_count(model):
    return sum(p.numel() for p in model.parameters())

# Cek bentuk tensor sebelum training.
with torch.no_grad():
    for name in ["ae", "sparse", "vae", "gan", "dcgan", "ddpm", "classifier", "frozen_decoder"]:
        model = make_model(name)
        count = sum(parameter_count(m) for m in model) if isinstance(model, tuple) else parameter_count(model)
        print(f"{name:16s}: {count:,} parameter")
        del model
