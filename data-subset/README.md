# Clothing Dataset (Balanced 1000)

## Overview
This dataset is prepared for image classification tasks.

- Total images: 1000
- Image format: `.jpg`
- Label file: `images.csv`
- Number of classes: 10
- Class balance: 100 images per class

## Folder Structure

```text
clothing-dataset/
  images.csv
  images/
    <image_id>.jpg
```

## CSV Schema (`images.csv`)

The CSV contains one row per image.

- `image`: image ID (without extension)
- `sender_id`: source/user ID metadata
- `label`: class label
- `kids`: boolean metadata (`True`/`False`)

Image path is formed as:

- `images/<image>.jpg`

## Class Distribution

Balanced classes (100 each):

- Dress
- Hat
- Longsleeve
- Not sure
- Outwear
- Pants
- Shirt
- Shoes
- Shorts
- T-Shirt

## Credits

Original dataset source:
https://github.com/alexeygrigorev/clothing-dataset

This balanced 1000-image version was derived from the original dataset above.
