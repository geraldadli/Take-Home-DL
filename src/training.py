from .runtime import *
from .data import *
from .models import *
def checkpoint_path(name, kind="best"):
    return RUN_DIR / "checkpoints" / f"{name}_{kind}.pt"

def save_checkpoint(state, path):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)

def load_model(name):
    path = checkpoint_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Latih {name} terlebih dahulu: {path}")
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = make_model(name)
    if isinstance(model, tuple):
        model[0].load_state_dict(checkpoint["G"])
        model[1].load_state_dict(checkpoint["D"])
        for m in model:
            m.eval()
    else:
        model.load_state_dict(checkpoint["model"])
        model.eval()
    return model

def capture_rng(generator):
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None}

def restore_rng(state, generator):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    generator.set_state(state["loader"].cpu())
    if DEVICE.type == "cuda" and state["cuda"] is not None:
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])

def objective(name, model, x, y, classifier=None):
    if name in ["ae", "sparse"]:
        rec, z = model(x)
        reconstruction = F.mse_loss(rec, x)
        sparsity = z.abs().sum(dim=1).mean()
        loss = reconstruction + (CFG["sparse_lambda"] * sparsity if name == "sparse" else 0)
        return loss, {"loss": loss.item(), "reconstruction": reconstruction.item(),
                      "l1_per_image": sparsity.item(), "active_fraction": (z > 1e-3).float().mean().item()}
    if name == "vae":
        rec, mu, logvar = model(x)
        # Jumlah BCE per citra, lalu rata-rata batch. KL menggunakan skala yang sama.
        bce = F.binary_cross_entropy(rec, x, reduction="sum") / len(x)
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
        loss = bce + CFG["vae_beta"] * kl
        return loss, {"loss": loss.item(), "bce": bce.item(), "kl": kl.item()}
    if name == "ddpm":
        loss = model.loss(x)
        return loss, {"loss": loss.item()}
    if name == "classifier":
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        return loss, {"loss": loss.item(), "accuracy": (logits.argmax(1) == y).float().mean().item()}
    if name == "frozen_decoder":
        with torch.no_grad():
            z = classifier.features(x)
        loss = F.mse_loss(model(z), x)
        return loss, {"loss": loss.item()}
    raise ValueError(name)

@torch.no_grad()
def validation_score(name, model, classifier=None):
    model.eval()
    total, count = 0.0, 0
    with isolated_seed(SEED + 101):
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss, details = objective(name, model, x, y, classifier)
            score = -details["accuracy"] if name == "classifier" else loss.item()
            total += score * len(x)
            count += len(x)
    return total / count

def train_one(name):
    seed_all()
    epochs = CFG["epochs"][name]
    history_path = RUN_DIR / "logs" / f"{name}.json"
    last_path = checkpoint_path(name, "last")
    generator = torch.Generator().manual_seed(SEED)
    loader = make_loader(train_set, shuffle=True, generator=generator)
    model = make_model(name)
    adversarial = isinstance(model, tuple)
    classifier = None
    if name == "frozen_decoder":
        classifier = load_model("classifier")
        classifier.requires_grad_(False)
    if adversarial:
        G, D = model
        optG = torch.optim.Adam(G.parameters(), lr=CFG["lr"][name], betas=(0.5, 0.999))
        optD = torch.optim.Adam(D.parameters(), lr=CFG["lr"][name], betas=(0.5, 0.999))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"][name])
    start, best, history = 0, float("inf"), []
    if last_path.exists():
        state = torch.load(last_path, map_location=DEVICE, weights_only=False)
        if state["epoch"] >= epochs:
            print(f"{name}: sudah selesai; checkpoint digunakan kembali.")
            return
        start, best, history = state["epoch"], state["best_score"], state["history"]
        if adversarial:
            G.load_state_dict(state["G"]); D.load_state_dict(state["D"])
            optG.load_state_dict(state["optG"]); optD.load_state_dict(state["optD"])
        else:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], generator)
        print(f"{name}: melanjutkan dari epoch {start}.")
    for epoch in range(start + 1, epochs + 1):
        if adversarial:
            G.train(); D.train()
        else:
            model.train()
        totals, seen = {}, 0
        synchronize(); started = time.perf_counter()
        for batch_index, (x, y) in enumerate(loader):
            if CFG["max_train_batches"] and batch_index >= CFG["max_train_batches"]:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            if adversarial:
                real = 2 * x - 1
                z = torch.randn(len(x), CFG["latent_gan"], device=DEVICE)
                fake = G(z)
                D.requires_grad_(True)
                optD.zero_grad(set_to_none=True)
                real_logits, fake_logits = D(real), D(fake.detach())
                # Logits + BCEWithLogits. Label smoothing hanya untuk target real.
                loss_D = 0.5 * (
                    F.binary_cross_entropy_with_logits(real_logits, torch.full_like(real_logits, 0.9)) +
                    F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
                loss_D.backward(); optD.step()
                D.requires_grad_(False)
                optG.zero_grad(set_to_none=True)
                generated_logits = D(fake)
                loss_G = F.binary_cross_entropy_with_logits(generated_logits, torch.ones_like(generated_logits))
                loss_G.backward(); optG.step()
                D.requires_grad_(True)
                details = {"G": loss_G.item(), "D": loss_D.item()}
            else:
                optimizer.zero_grad(set_to_none=True)
                loss, details = objective(name, model, x, y, classifier)
                loss.backward(); optimizer.step()
            if not all(np.isfinite(v) for v in details.values()):
                raise FloatingPointError(f"Loss non-finite pada {name}, epoch {epoch}.")
            for key, value in details.items():
                totals[key] = totals.get(key, 0.0) + value * len(x)
            seen += len(x)
        synchronize(); train_seconds = time.perf_counter() - started
        row = {"epoch": epoch, "train_seconds": train_seconds,
               "training_device": ENVIRONMENT["gpu"] or "CPU",
               **{k: v / seen for k, v in totals.items()}}
        # GAN: gunakan epoch terakhir yang ditetapkan sebelumnya, tanpa memilih memakai test FID.
        score = -epoch if adversarial else validation_score(name, model, classifier)
        if not adversarial:
            row["validation_score"] = score
        improved = score < best
        best = min(best, score)
        history.append(row)
        common = {"epoch": epoch, "best_score": best, "history": history,
                  "rng": capture_rng(generator), "config": CFG}
        if adversarial:
            state = {**common, "G": G.state_dict(), "D": D.state_dict(),
                     "optG": optG.state_dict(), "optD": optD.state_dict()}
        else:
            state = {**common, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
        if improved:
            # Checkpoint best hanya menyimpan bobot dan informasi pemilihan.
            selected = {k: v for k, v in state.items() if k not in ["optimizer", "optG", "optD", "rng"]}
            save_checkpoint(selected, checkpoint_path(name))
        save_checkpoint(state, last_path)
        history_path.write_text(json.dumps(history, indent=2))
        print(name, f"epoch {epoch}/{epochs}",
              {k: round(v, 4) if isinstance(v, (float, int)) else v for k, v in row.items()})
    del model, classifier
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

# Jalankan tiap model di sel terpisah pada notebook agar kemajuan mudah dipantau.
