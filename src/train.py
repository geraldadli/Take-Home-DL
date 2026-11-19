from .runtime import *
from .data import *
from .models import *
from .training import *

if __name__ == "__main__":
    for name in ["classifier", "frozen_decoder", "ae", "sparse", "vae", "gan", "dcgan", "ddpm"]:
        train_one(name)
