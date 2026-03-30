from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ImageDataset(Dataset):
    def __init__(self, root: str, image_size: int = 128, train: bool = True):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"No existeix la carpeta: {self.root}")

        self.files = sorted(
            [p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        )
        if len(self.files) == 0:
            raise RuntimeError(f"No he trobat imatges a {self.root}")

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)
