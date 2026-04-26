# CNN-Based Clothing Classification

This project is a demonstration of building, training and evaluating a Convolutional Neural Network for classifying clothing images, as well as saving and testing those models on a test dataset. It compares using deeper CNN with 2 vs 4 convolutional layers, and utilizing augmentation.

COSC4368 @ UH

## Information

This project sources its data from [clothing-dataset-small](https://github.com/alexeygrigorev/clothing-dataset-small) from [alexeygrigorev](https://github.com/alexeygrigorev). All credits for sourcing the images go to him and contributors. 

There are 10 categories

- T-shirt
- Hat
- Dresse
- Outwear
- Shorts
- Pants
- Longsleeve
- Shoes
- Shirt
- Not sure (this is project-optional, but included for fun)

Each category *should* store up to 100 images each, making up **1000** images total.

The project data is split into the following divisions.

```python
n = len(df) # split 70% train, 15% validation, 15% test
train_df = df.iloc[:int(0.7 * n)]
val_df = df.iloc[int(0.7 * n):int(0.85 * n)]
test_df = df.iloc[int(0.85 * n):]
```

## Dependencies

In order to run this project, you must also install **Make** and **pip** on your system.

This project utilizied [PyTorch](https://pytorch.org/get-started/locally/). It is recommended to follow the installation instructions found directly on the get-started page of pytorch.org.

This project is also dependent on python packages `pandas, scikit-learn` and `matplotlib`.

- To install dependencies, run `make install`.

- To run the project and train the models, run `make run`.

- To evaluate existing models after training, run `make eval`.

If you don't want to install **Make**, you can just run these commands independently. Look at **Makefile** for how those commands are executed.

## Instructions

To get the best out of this project, you *must* train your models first. It will generate a total of 4 models w/the following properties.

- not_deep_noaug
- deep_noaug
- not_deep_aug
- deep_aug

where
- deep means there are 4 conv layers, not_deep means there are 2
- aug means the training data has been augmented/transformed

Look at `src/globalVariables.py` for hyrprparameters to change.

Once you have decided your hyprparameters, simply run `make run` to start training.
 
```python
EPOCHS = 60
LEARNING_RATE = 0.0001
BATCH_SIZE=32
KERNEL_SIZE=3
EVAL_MODELS_PATH=os.path.join(os.getcwd(), "models") # change "models" to "example-models" to try the provided example models during eval
```

Once training has finished, they will end up in a `models/` folder. You can now run `make eval`, select your model, and see it evaluated against the test set!

If you want to run the **example-models**, simply go to `src/globalVariables.py` and change EVAL_MODELS_PATH "models" to "example-models".


