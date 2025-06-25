Installing dependencies
```Bash
pip3 install torch torchvision
pip3 install tensorboard thop spikingjelly==0.0.0.0.14 cupy-cuda11x timm
```

`utils\Data_processing_raw_classification.py`: Code for event representation processing.

|-- `models\SpikingResformer.py`: Code for tasks using SRF

|-- `models\Alexnet+LSTM.py`: Code for tasks using AlexNet+LSTM

|-- `models\ResNet34_EST.py`: Code for tasks using ResNet34+EST

For the implementation of training code for regression and classification tasks, please refer to the code in `train_regression.py` and `train_classification.py`. Additionally, for dataset loading procedures, consult the code provided in `dataset.py`.



