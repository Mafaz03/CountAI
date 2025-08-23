import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torch.utils.data import WeightedRandomSampler

from torchvision import transforms


# 1 - 49: Not Defective
# 50 - 61: Defective

# Image Dataset Loader
class Image_dataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.list_files = os.listdir(root_dir)
    
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.5, 0.5, 0.5)
        ])

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        image_path = self.list_files[index]
        file_path = os.path.join(self.root_dir, image_path)
        image = Image.open(file_path)
        label = "Defective" if int(image_path.split('.jpg')[0]) >= 50 else "Not Defective"

        if self.transform:
            image = self.transform(image)
        
        return np.array(image), label
    
    