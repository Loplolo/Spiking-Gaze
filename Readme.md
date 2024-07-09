# Usage

General usage: 

```python main.py [-h] {nengo, keras, converted, nengo_alt} {train,eval,webcam,show} [--dataset_dir DATASET_DIR] [--train_split TRAIN_SPLIT] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--save SAVE] [--load LOAD]```

Example:

```python3 main.py converted train --dataset_dir .\dataset\MPIIFaceGaze --save model.keras```

## Types
types = ['nengo', 'keras', 'converted', 'nengo_alt']

- Nengo = model converted first from keras AlexNet implementation and then trained
- Keras = AlexNet implementation with keras
- Converted = model converted from trained keras AlexNet implementation
- Nengo_alt = WIP

## Actions
actions = ['train', 'eval', 'webcam', 'show']

- train = fit the model with given dataset
- eval = evaluate the model with given dataset
- webcam = start webcam loop for testing
- show = show predictions side by side with images
