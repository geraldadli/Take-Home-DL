from .runtime import *
from .data import *
from .models import *
from .training import *
@torch.no_grad()
def reconstruction_metrics(reconstruct, loader=test_loader):
    sums = np.zeros(3, dtype=np.float64)
    count = 0
    for x, _ in loader:
        x = x.to(DEVICE)
        rec = reconstruct(x).clamp(0, 1)
        per_image_mse = (rec - x).square().flatten(1).mean(1)
        per_image_psnr = -10 * torch.log10(per_image_mse.clamp_min(1e-12))
        ssim = structural_similarity_index_measure(rec, x, data_range=1.0)
        sums += [per_image_mse.sum().item(), per_image_psnr.sum().item(), ssim.item() * len(x)]
        count += len(x)
    return dict(zip(["test_mse", "test_psnr_db", "test_ssim"], (sums / count).tolist()))

@torch.no_grad()
def classifier_accuracy(model, loader):
    correct, total = 0, 0
    for x, y in loader:
        prediction = model(x.to(DEVICE)).argmax(1).cpu()
        correct += (prediction == y).sum().item()
        total += len(y)
    return correct / total

@torch.no_grad()
def generate(name, model, n, z=None):
    if name == "ddpm":
        return model.sample(n, initial_noise=z)
    dimension = CFG["latent_ae"] if name == "vae" else CFG["latent_gan"]
    z = torch.randn(n, dimension, device=DEVICE) if z is None else z
    return model.decoder(z) if name == "vae" else ((model(z) + 1) / 2).clamp(0, 1)

@torch.no_grad()
def latency_measurement(name, model):
    # Timer mencakup inferensi model; tidak mencakup transfer data dan penulisan berkas.
    if name in ["ae", "sparse"]:
        x = torch.stack([test_set[i][0] for i in range(64)]).to(DEVICE)
        call = lambda: model(x)[0]
        operation = "encode + decode, batch 64"
    else:
        shape = (64, 1, 28, 28) if name == "ddpm" else (64, CFG["latent_ae"] if name == "vae" else CFG["latent_gan"])
        z = torch.randn(*shape, device=DEVICE)
        call = lambda: generate(name, model, 64, z)
        operation = "generation, batch 64"
    warmups, repeats = ((1, 2) if PROFILE == "smoke" else ((2, 5) if name == "ddpm" else (5, 30)))
    for _ in range(warmups):
        call()
    times = []
    for _ in range(repeats):
        synchronize(); start = time.perf_counter()
        call()
        synchronize(); times.append((time.perf_counter() - start) * 1000)
    return {"latency_ms_batch64": float(np.median(times)), "latency_p95_ms": float(np.percentile(times, 95)),
            "latency_repeats": repeats, "latency_operation": operation}

@torch.no_grad()
def calculate_fid(name, model):
    if not CFG["calculate_fid"]:
        return None
    from torchmetrics.image.fid import FrechetInceptionDistance
    # Standard FID: Inception-v3, 2048 fitur, grayscale diulang menjadi RGB.
    # Tidak mengganti extractor dengan classifier Fashion-MNIST.
    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    metric.set_dtype(torch.float64)
    n = min(CFG["fid_samples"], len(test_set))
    loader = make_loader(Subset(test_set, list(range(n))), batch_size=64)
    for x, _ in loader:
        metric.update(x.to(DEVICE).repeat(1, 3, 1, 1), real=True)
    # Cache sampel per batch: generation DDPM dapat dilanjutkan jika Colab terputus.
    # Hash checkpoint mencegah penggunaan cache dari bobot model yang berbeda.
    digest = hashlib.sha256()
    with checkpoint_path(name).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    cache_dir = RUN_DIR / "samples" / "fid_cache" / f"{name}_{digest.hexdigest()[:12]}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, n, 64):
        cache_path = cache_dir / f"batch_{start:05d}.pt"
        if cache_path.exists():
            fake = torch.load(cache_path, map_location=DEVICE, weights_only=True)
        else:
            with isolated_seed(SEED + 202 + start):
                fake = generate(name, model, min(64, n - start))
            save_checkpoint(fake.cpu(), cache_path)
        assert len(fake) == min(64, n - start)
        metric.update(fake.repeat(1, 3, 1, 1), real=False)
        if name == "ddpm" and (start // 64) % 10 == 0:
            print(f"FID DDPM: {start + len(fake)}/{n} sampel")
    score = float(metric.compute().cpu())
    del metric
    return score
