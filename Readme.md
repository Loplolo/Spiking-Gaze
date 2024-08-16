# Introduction

Demonstration of the use of Spiking Neural Network for 3D Gaze Estimation task using convolutional neural networks

# Requirements

Tested with python 3.10.14
Install Python libraries with ```pip install -r requirements.txt```

MPIIFaceGaze dataset: https://perceptualui.org/files/datasets/MPIIFaceGaze.zip ( unzip in ./dataset/ )

# Usage

General usage: 

```console 
python main.py [-h] {ann, snn} {train,eval,webcam,show} [--dataset_dir DATASET_DIR] [--calib_path CAMERA_CALIB_FILE] [--train_split TRAIN_SPLIT] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LEARNING_RATE] [--save SAVE] [--load LOAD] [--n_steps N_STEPS] [--synapse SYNAPSE] [--sfr SCALE_FIRING_RATE]
```

Example:

```console
python3 main.py ann train --dataset_dir .\dataset\MPIIFaceGaze --save model_01
```

## Types
types = ['ann', 'snn']

- ann = Run/Train an Artificial Neural Network inside a Nengo_DL network
- snn = Run the converted Spiking Neural Network inside a Nengo_DL network

## Actions
actions = ['train', 'eval', 'webcam', 'show']

- train = fit the model with given dataset
- eval = evaluate the model with given dataset
- webcam = start webcam loop for testing
- show = show predictions side by side with images

# Record Dataset

Usage: 
```python3 record_dataset.py [-h] [--dataset_dir DATASET_DIR] [--id ID_PERSON] [--day DAY_FOLDER_NAME]```

Example:
```python3 record_dataset.py --dataset_dir ./dataset/MPIIFaceGaze --id p00 --day day01```

Filesystem structure:
<pre>
📦p00
 ┣ 📂Calibration
 ┃ ┣ 📜Camera.mat
 ┃ ┣ 📜monitorPose.mat
 ┃ ┗ 📜screenSize.mat
 ┣ 📂day00
 ┃ ┣ 📜0001.jpg
 ┃ ┣ ...
 ┃ ┗ 📜annotations.txt
 ┣ 📂day01
 ┃ ┣ 📜001.jpg
 ┃ ┣ ...
 ┃ ┗ 📜annotations.txt
 ┗ 📂day02
 ┃ ┣ 📜0001.jpg
 ┃ ┣ ...
 ┃ ┗ 📜annotations.txt
</pre>
