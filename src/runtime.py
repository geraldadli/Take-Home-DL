import os, sys, json, time, random, hashlib, platform, csv
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
from torchmetrics.functional.image import structural_similarity_index_measure

# Jalankan smoke dahulu; ubah menjadi "full" untuk mengumpulkan hasil tugas.
PROFILE = os.environ.get("STUDY_PROFILE", "full")
USE_GOOGLE_DRIVE = False
SEED = 42
BATCH_SIZE = 128
EPOCH_OVERRIDE = {}  # Contoh: {"ae": 30, "ddpm": 50}. Pilih menggunakan validation set.

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = Path("/content/drive/MyDrive/GenerativeModelsStudy")
else:
    BASE_DIR = Path.cwd() / "generative_models_runs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(min(4, os.cpu_count() or 1))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_all()
CFG = {
    "profile": PROFILE, "seed": SEED, "batch_size": BATCH_SIZE,
    "latent_ae": 64, "latent_gan": 100, "sparse_lambda": 1e-4,
    "vae_beta": 1.0, "diffusion_steps": 1000, "unet_channels": 32,
    "epochs": {"ae": 20, "sparse": 20, "vae": 30, "gan": 40,
               "dcgan": 40, "ddpm": 30, "classifier": 20, "frozen_decoder": 20},
    "lr": {"ae": 1e-3, "sparse": 1e-3, "vae": 1e-3, "gan": 2e-4,
           "dcgan": 2e-4, "ddpm": 2e-4, "classifier": 1e-3,
           "frozen_decoder": 1e-3},
    "fid_samples": 7000, "calculate_fid": True, "max_train_batches": None,
}
if PROFILE == "smoke":
    CFG.update(diffusion_steps=20, unet_channels=16, fid_samples=128,
               calculate_fid=False, max_train_batches=2)
    CFG["epochs"] = {k: 1 for k in CFG["epochs"]}
elif PROFILE != "full":
    raise ValueError("PROFILE harus 'full' atau 'smoke'.")
CFG["epochs"].update(EPOCH_OVERRIDE)
RUN_ID = hashlib.sha256(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:8]
RUN_DIR = BASE_DIR / f"{PROFILE}_fashionmnist_seed{SEED}_{RUN_ID}"
for folder in ["checkpoints", "figures", "logs", "metrics", "samples"]:
    (RUN_DIR / folder).mkdir(parents=True, exist_ok=True)
(RUN_DIR / "config.json").write_text(json.dumps(CFG, indent=2))
ENVIRONMENT = {
    "python": sys.version, "torch": torch.__version__, "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
    "platform": platform.platform(), "profile": PROFILE,
}
(RUN_DIR / "environment.json").write_text(json.dumps(ENVIRONMENT, indent=2))
print(json.dumps(ENVIRONMENT, indent=2))
print("Folder hasil:", RUN_DIR)
if PROFILE == "smoke":
    print("SMOKE TEST: hasil hanya untuk debugging, bukan hasil laporan.")
if DEVICE.type == "cpu" and PROFILE == "full":
    print("Aktifkan GPU Colab (Runtime > Change runtime type) sebelum training penuh.")

@contextmanager
def isolated_seed(seed):
    devices = [torch.cuda.current_device()] if DEVICE.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield

def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
