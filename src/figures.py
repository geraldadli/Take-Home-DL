from .runtime import *
from .data import *
from .models import *
from .training import *
from .metrics import *
from .mlp_analysis import *
def finish_figure(fig, number, title):
    status = " | SMOKE TEST: BUKAN HASIL LAPORAN" if PROFILE == "smoke" else ""
    fig.suptitle(f"Figur {number}. {title}{status}", fontsize=13)
    fig.savefig(RUN_DIR / "figures" / f"figure_{number}.png", dpi=240, bbox_inches="tight")
    fig.savefig(RUN_DIR / "figures" / f"figure_{number}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

def show_grid(ax, batch, nrow=4):
    grid = make_grid(batch.detach().cpu(), nrow=nrow, padding=2, pad_value=1)
    ax.imshow(grid.permute(1, 2, 0).clamp(0, 1).numpy())
    ax.axis("off")

display_x = torch.stack([test_set[i][0] for i in range(8)]).to(DEVICE)
display_y = torch.tensor([int(test_set[i][1]) for i in range(8)], device=DEVICE)
models = {}
for name in ["ae", "sparse", "vae", "gan", "dcgan", "ddpm"]:
    m = load_model(name)
    models[name] = m[0] if isinstance(m, tuple) else m

# Figur 1: pasangan citra uji yang sama untuk seluruh rekonstruksi.
with torch.no_grad():
    rows = [display_x, models["ae"](display_x)[0], models["sparse"](display_x)[0],
            models["vae"](display_x, sample=False)[0]]
fig, axes = plt.subplots(4, 8, figsize=(12, 6), constrained_layout=True)
for r, (name, batch) in enumerate(zip(["Asli", "Vanilla AE", "Sparse AE", "VAE (mu)"], rows)):
    for c in range(8):
        axes[r, c].imshow(batch[c, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if c == 0:
            axes[r, c].set_ylabel(name)
finish_figure(fig, 1, "Perbandingan rekonstruksi")

# Figur 2: seed tetap, tidak memilih hanya sampel terbaik.
generated = {}
fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
for ax, name in zip(axes, ["vae", "gan", "dcgan", "ddpm"]):
    with isolated_seed(SEED + 404), torch.no_grad():
        generated[name] = generate(name, models[name], 16).cpu()
    show_grid(ax, generated[name])
    ax.set_title(name.upper())
torch.save(generated, RUN_DIR / "samples" / "random_generation.pt")
finish_figure(fig, 2, "Random sampling: 16 citra per model")

# Figur 3: seluruh tahap baseline classifier, termasuk kontrol AE.
with torch.no_grad():
    projected_class = transpose_from_class(display_y)
    projected_latent = transpose_from_features(display_x)
    decoded_frozen = frozen_decoder(classifier.features(display_x))
    decoded_ae = models["ae"](display_x)[0]
fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
show_grid(axes[0, 0], display_x, 8)
axes[0, 0].set_title("A. Citra asli")
show_grid(axes[0, 1], torch.cat([projected_class, projected_latent]), 8)
axes[0, 1].set_title("B. Transpose kelas (atas), transpose fitur (bawah)")
show_grid(axes[1, 0], torch.cat([decoded_frozen, decoded_ae]), 8)
axes[1, 0].set_title("C. Frozen decoder (atas), Vanilla AE (bawah)")
show_grid(axes[1, 1], mlp_samples_test, 4)
axes[1, 1].set_title("D. Sampling Gaussian test-fit: diagnostik saja")
finish_figure(fig, 3, "Eksperimen bobot dan fitur classifier")
fig, ax = plt.subplots(figsize=(4, 4))
show_grid(ax, mlp_samples_train)
ax.set_title("Kontrol sampling: statistik train set")
fig.savefig(RUN_DIR / "figures" / "classifier_train_fit_control.png", dpi=240, bbox_inches="tight")
plt.show(); plt.close(fig)

# Figur 4: AE/VAE memakai citra uji; GAN tidak memiliki encoder.
index_a = 0
label_a = int(test_set[index_a][1])
index_b = next(i for i in range(1, len(test_set)) if int(test_set[i][1]) != label_a)
label_b = int(test_set[index_b][1])
pair = torch.stack([test_set[index_a][0], test_set[index_b][0]]).to(DEVICE)
alphas = torch.tensor([0., .2, .4, .6, .8, 1.], device=DEVICE)[:, None]
interpolation_info = {"test_indices": [int(test_idx[index_a]), int(test_idx[index_b])],
                      "target_classes": [label_a, label_b], "gan_endpoints": {}}
fig, axes = plt.subplots(5, 6, figsize=(10, 8), constrained_layout=True)
for r, name in enumerate(["ae", "sparse", "vae", "gan", "dcgan"]):
    model = models[name]
    with torch.no_grad(), isolated_seed(SEED + 505):
        if name in ["ae", "sparse"]:
            endpoints = model.encoder(pair)
        elif name == "vae":
            endpoints = model.encode(pair)[0]
        else:
            # Cari prior samples yang diprediksi classifier sebagai dua kelas target.
            candidates = torch.randn(512, CFG["latent_gan"], device=DEVICE)
            candidate_images = generate(name, model, len(candidates), candidates)
            scores = classifier(candidate_images).softmax(1)
            chosen = []
            for label in [label_a, label_b]:
                valid = torch.where(scores.argmax(1) == label)[0]
                selected = valid[scores[valid, label].argmax()] if len(valid) else scores[:, label].argmax()
                chosen.append(int(selected))
            endpoints = candidates[chosen]
            actual = scores[chosen].argmax(1).tolist()
            interpolation_info["gan_endpoints"][name] = {"predicted_classes": actual,
                "target_classes_found": actual == [label_a, label_b],
                "note": "Class labels are classifier predictions, not guaranteed conditioning."}
            if actual != [label_a, label_b]:
                print(name, "tidak menemukan kedua kelas target; periksa kemungkinan mode dropping.")
        path = (1 - alphas) * endpoints[0] + alphas * endpoints[1]
        decoded = model.decoder(path) if name in ["ae", "sparse", "vae"] else generate(name, model, 6, path)
    for c in range(6):
        axes[r, c].imshow(decoded[c, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == 0:
            axes[r, c].set_title(f"alpha = {alphas[c].item():.1f}")
        if c == 0:
            axes[r, c].set_ylabel(name.upper())
(RUN_DIR / "metrics" / "interpolation_protocol.json").write_text(json.dumps(interpolation_info, indent=2))
finish_figure(fig, 4, "Interpolasi laten pada lima model")

# Figur 5: loss antarmodel memiliki satuan berbeda; tidak dibandingkan magnitudonya.
fig = plt.figure(figsize=(13, 15), constrained_layout=True)
layout = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, .65])
plot_keys = {"ae": ["loss"], "sparse": ["loss", "reconstruction"], "vae": ["loss", "bce", "kl"],
             "gan": ["G", "D"], "dcgan": ["G", "D"], "ddpm": ["loss"]}
for i, name in enumerate(["ae", "sparse", "vae", "gan", "dcgan", "ddpm"]):
    ax = fig.add_subplot(layout[i // 2, i % 2])
    history = json.loads((RUN_DIR / "logs" / f"{name}.json").read_text())
    for key in plot_keys[name]:
        ax.plot([h["epoch"] for h in history], [h[key] for h in history], marker=".", label=key)
    ax.set(title=name.upper(), xlabel="Epoch", ylabel="Loss (skala model masing-masing)")
    ax.grid(alpha=.25); ax.legend()
with isolated_seed(SEED + 606):
    _, states = models["ddpm"].sample(1, trajectory=True)
trajectory_layout = layout[3, :].subgridspec(1, len(states))
for i, (step, image) in enumerate(states.items()):
    ax = fig.add_subplot(trajectory_layout[0, i])
    ax.imshow(((image[0, 0] + 1) / 2).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"x_{step}"); ax.axis("off")
torch.save(states, RUN_DIR / "samples" / "ddpm_trajectory.pt")
finish_figure(fig, 5, "Kurva training dan reverse diffusion")

print("Lima figur tersimpan di", RUN_DIR / "figures")
