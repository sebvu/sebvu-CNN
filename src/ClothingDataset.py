from torchvision import transforms
from torch.utils.data import Dataset 
from PIL import Image

class ClothingDataset(Dataset):
    def __init__(self, df, labels, is_augmented: bool = False):
        self.df = df
        self.labels = labels

        if not is_augmented: 
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        else:
            self.transform = transforms.Compose([
                # resize all images to 128x128, dataset images may have varied resolutions and the networks expects a fixed resolution
                transforms.Resize((128, 128)), 
                # random augmentations for training
                transforms.RandomRotation(45),
                transforms.RandomHorizontalFlip(),
                # covert PIL image to python tensor shape of [3, 128, 128]. rescale pixel values from 0-255 int to 0-1 floats
                # channel ordered as [R, G, B]
                transforms.ToTensor(), 
                # normalize using mean=0.5, std=0.5
                # zero-centered -1 > 1 data trains more stably
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        row_image_id = row["image"]
        row_label = row["label"]
        img = Image.open(f"data/images/{row_image_id}.jpg")

        img_normalized_tensor = self.transform(img)

        return (img_normalized_tensor, self.labels[row_label])
