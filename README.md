# CNN-Based Clothing Classification

This project is a demonstration of building, training and evaluating a Convolutional Neural Network for classifying clothing images, as well as saving and testing those models on a test dataset. It compares using deeper CNN with 2 vs 4 convolutional layers, and utilizing augmentation.

## Sourced Data

This project sources its data from [clothing-dataset-small](https://github.com/alexeygrigorev/clothing-dataset-small) from [alexeygrigorev](https://github.com/alexeygrigorev). All credits for sourcing the images go to him and contributors.

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
 
```python
EPOCHS = 60
LEARNING_RATE = 0.0001
BATCH_SIZE=32
KERNEL_SIZE=3
EVAL_MODELS_PATH=os.path.join(os.getcwd(), "models") # change "models" to "example-models" to try the provided example models during eval
```

Once these have been generated, you can then run `make eval` to test those models out directly and have the results outputted to console.

