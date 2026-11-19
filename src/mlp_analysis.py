from .runtime import *
from .data import *
from .models import *
from .training import *
from .metrics import *
classifier = load_model("classifier")
classifier.requires_grad_(False)
frozen_decoder = load_model("frozen_decoder")
ae_reference = load_model("ae")

@torch.no_grad()
def transpose_from_class(y):
    one_hot = F.one_hot(y, 10).float()
    # Batch berbentuk baris: y @ W3 setara W3^T y dalam notasi vektor kolom.
    z = F.relu(one_hot @ classifier.fc3.weight)
    h = F.relu(z @ classifier.fc2.weight)
    return torch.sigmoid(h @ classifier.fc1.weight).view(-1, 1, 28, 28)

@torch.no_grad()
def transpose_from_features(x):
    z = classifier.features(x)
    h = F.relu(z @ classifier.fc2.weight)
    return torch.sigmoid(h @ classifier.fc1.weight).view(-1, 1, 28, 28)

@torch.no_grad()
def latent_statistics(split):
    # Statistik dihitung tanpa mengubah bobot classifier atau decoder.
    total = torch.zeros(64, dtype=torch.float64)
    total_square = torch.zeros_like(total)
    count = 0
    for x, _ in make_loader(split):
        z = classifier.features(x.to(DEVICE)).cpu().double()
        total += z.sum(0); total_square += z.square().sum(0)
        count += len(z)
    mean = total / count
    std = (total_square / count - mean.square()).clamp_min(0).sqrt()
    return mean.float().to(DEVICE), std.float().to(DEVICE)

mlp_metrics = {
    "latent_transpose": reconstruction_metrics(transpose_from_features),
    "frozen_latent_decoder": reconstruction_metrics(lambda x: frozen_decoder(classifier.features(x))),
    "vanilla_ae": reconstruction_metrics(lambda x: ae_reference(x)[0]),
}
gram_checks = {}
for layer_name in ["fc1", "fc2", "fc3"]:
    weight = getattr(classifier, layer_name).weight.detach().cpu().double()
    eye = torch.eye(weight.shape[1], dtype=torch.float64)
    gram_checks[layer_name] = {
        "weight_shape": list(weight.shape), "rank_upper_bound": min(weight.shape),
        "input_dimension": weight.shape[1],
        "relative_gram_error": float(torch.linalg.norm(weight.T @ weight - eye) / torch.linalg.norm(eye))}
mu_train, std_train = latent_statistics(train_set)
mu_test, std_test = latent_statistics(test_set)
with isolated_seed(SEED + 303), torch.no_grad():
    eps = torch.randn(16, 64, device=DEVICE)
    mlp_samples_train = frozen_decoder(mu_train + std_train * eps).cpu()
    mlp_samples_test = frozen_decoder(mu_test + std_test * eps).cpu()
torch.save({"train_mu": mu_train.cpu(), "train_std": std_train.cpu(),
            "test_mu": mu_test.cpu(), "test_std": std_test.cpu(),
            "train_fit_samples": mlp_samples_train, "test_fit_samples": mlp_samples_test},
           RUN_DIR / "samples" / "classifier_sampling.pt")
(RUN_DIR / "metrics" / "mlp_experiments.json").write_text(json.dumps({
    "profile": PROFILE, "reconstruction": mlp_metrics, "gram_checks": gram_checks,
    "sampling_protocol": "Train-fit is the clean sampling baseline. Test-fit follows task stage 5, is diagnostic only, and is not an independent held-out evaluation.",
    "interpretation": "Frozen decoder reconstruction or Gaussian sampling may succeed partially; report observed results, not an assumed failure."}, indent=2))
print(json.dumps(mlp_metrics, indent=2))
print("Statistik test-fit disimpan sebagai diagnostik sesuai soal; tidak dipakai untuk tuning.")
