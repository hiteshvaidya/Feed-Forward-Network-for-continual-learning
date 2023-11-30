# Feed-Forward-Network-for-continual-learning
This repository contains codebase for a basic Feed Forward Network that operates in a continual learning setting. The code is developed from scratch without using keras library. This can also be adapted to build a vanilla feed forward neural network that is trained in a normal i.i.d. setting.

Please refer to this [blog](https://medium.com/analytics-vidhya/how-to-write-a-neural-network-in-tensorflow-from-scratch-without-using-keras-e056bb143d78) for a step-by-step guide.

## Installation
Clone this repository
```
git clone git@github.com:hiteshvaidya/Feed-Forward-Network-for-continual-learning.git
```

Create a new environment with python 3.10
```
conda create -n lifelong python=3.10
conda activate lifelong
```

Install all the necessary libraries
```
pip install -r requirements.txt
```

## Structure
```
lifelong
├── data
│   ├── cifar10
│   ├── fashion
│   ├── iid
│   ├── kmnist
│   └── mnist
├── requirements.txt
└── src
    ├── __pycache__
    ├── cifarConv.py
    ├── layer.py
    ├── main.py
    ├── network.py
    └── util.py
```

## Launching experiments
```
python main.py -d mnist -n 20 -lr 0.01 -b 32
```
