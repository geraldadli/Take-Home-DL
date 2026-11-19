from .runtime import *
# Gabungkan 70.000 citra, kemudian buat split baru 80:10:10 sesuai soal.
# Split ini berbeda dari split resmi Fashion-MNIST 60.000/10.000.
DATA_DIR = (Path("/content/data") if IN_COLAB else BASE_DIR / "data")
original_train = FashionMNIST(DATA_DIR, train=True, download=True)
original_test = FashionMNIST(DATA_DIR, train=False, download=True)
images = torch.cat([original_train.data, original_test.data])
labels = torch.cat([original_train.targets, original_test.targets])

class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return self.images[index].float().unsqueeze(0) / 255.0, self.labels[index]

dataset = ImageDataset(images, labels)
indices = np.arange(len(dataset))
train_idx, heldout_idx = train_test_split(
    indices, test_size=0.2, stratify=labels.numpy(), random_state=SEED)
val_idx, test_idx = train_test_split(
    heldout_idx, test_size=0.5, stratify=labels.numpy()[heldout_idx], random_state=SEED)
assert not (set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx)
            or set(val_idx) & set(test_idx))
np.savez(RUN_DIR / "split_indices.npz", train=train_idx, val=val_idx, test=test_idx)
if PROFILE == "smoke":
    train_idx, val_idx, test_idx = train_idx[:1024], val_idx[:256], test_idx[:256]
train_set, val_set, test_set = [Subset(dataset, idx) for idx in [train_idx, val_idx, test_idx]]
np.savez(RUN_DIR / "used_indices.npz", train=train_idx, val=val_idx, test=test_idx)

def make_loader(split, shuffle=False, batch_size=BATCH_SIZE, generator=None):
    # num_workers=0 menjaga notebook mudah dijalankan di Colab dan Windows.
    return DataLoader(split, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=DEVICE.type == "cuda", generator=generator)

val_loader = make_loader(val_set)
test_loader = make_loader(test_set)
CLASS_NAMES = original_train.classes
print("Train / validation / test:", len(train_set), len(val_set), len(test_set))
print("Semua model menggunakan indeks citra yang sama.")
