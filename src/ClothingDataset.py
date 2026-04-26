import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset 
from PIL import Image

class ClothingDataset(Dataset):
    def __init__(self, df, labels, is_augmented: bool = False):
        self.df = df
        self.labels = labels

        """
            Augmentation techniques applied to Dataset will only be applied if the is_augmented flag is true. These are only true for the models that require is_augmented.
        """
        if not is_augmented: 
            self.transform = v2.Compose([
                v2.Resize((128, 128)), 
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])
        else:
            self.transform = v2.Compose([
                # resize all images to 128x128, dataset images may have varied resolutions and the networks expects a fixed resolution
                v2.Resize((128, 128)), 
                # random augmentations for training
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                v2.RandomAffine(degrees=[-5, 5], translate=(0.05, 0.05)),
                # covert PIL image to python tensor shape of [3, 128, 128]. rescale pixel values from 0-255 int to 0-1 floats
                # channel ordered as [R, G, B]
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                # normalize using mean=0.5, std=0.5
                # zero-centered -1 > 1 model ignores scale through normalization and looks for patterns only
                v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        row_image_id = row["image"]
        row_label = row["label"]
        img = Image.open(f"data/images/{row_image_id}.jpg")

        img_normalized_tensor = self.transform(img)

        return (img_normalized_tensor, self.labels[row_label])
