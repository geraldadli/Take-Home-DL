from .runtime import *
from .data import *
from .models import *
from .training import *
from .metrics import *

metrics = []
for name in ["ae", "sparse", "vae", "gan", "dcgan", "ddpm"]:
    model = load_model(name)
    if isinstance(model, tuple):
        model = model[0]
    model.eval()
    history = json.loads((RUN_DIR / "logs" / f"{name}.json").read_text())
    row = {"model": name, "profile": PROFILE, "parameters_inference_model": parameter_count(model),
           "evaluation_device": ENVIRONMENT["gpu"] or "CPU",
           "training_devices": "; ".join(sorted({h.get("training_device", "not recorded") for h in history})),
           "latent_dim": "1x28x28 noise" if name == "ddpm" else (64 if name in ["ae", "sparse", "vae"] else 100),
           "training_seconds_per_epoch": float(np.mean([h["train_seconds"] for h in history])),
           "epochs_run": len(history), "test_mse": None, "test_psnr_db": None, "test_ssim": None,
           "fid": None, "fid_n_real": None, "fid_n_fake": None,
           "stability_observation": "ISI setelah meninjau kurva dan grid sampel"}
    if name in ["ae", "sparse"]:
        row.update(reconstruction_metrics(lambda x: model(x)[0]))
    elif name == "vae":
        row.update(reconstruction_metrics(lambda x: model(x, sample=False)[0]))
    row.update(latency_measurement(name, model))
    if name in ["vae", "gan", "dcgan", "ddpm"]:
        row["fid"] = calculate_fid(name, model)
        if row["fid"] is not None:
            row["fid_n_real"] = row["fid_n_fake"] = min(CFG["fid_samples"], len(test_set))
    metrics.append(row)
    print(name, row)
    # Simpan per model agar kemajuan evaluasi tidak hilang.
    (RUN_DIR / "metrics" / "comparison.json").write_text(json.dumps(metrics, indent=2))
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
with (RUN_DIR / "metrics" / "comparison.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(metrics[0]))
    writer.writeheader(); writer.writerows(metrics)

classifier = load_model("classifier")
accuracy = classifier_accuracy(classifier, test_loader)
print(f"Akurasi test classifier: {accuracy:.2%}; syarat tugas: >=88%.")
if accuracy < 0.88:
    print("Syarat belum terpenuhi. Gunakan validation set untuk memilih perbaikan; jangan tuning pada test set.")
(RUN_DIR / "metrics" / "classifier.json").write_text(json.dumps({
    "test_accuracy": accuracy, "meets_88_percent": accuracy >= .88, "profile": PROFILE}, indent=2))
