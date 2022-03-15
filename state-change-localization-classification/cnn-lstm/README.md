# EGO4D Hands and Objects Benchmark: code for training and evaluating the Bi-directional LSTM baseline

This is the github repository containing the code for reproducing the Bi-directional LSTM reults in the Hands and Objects Benchmark, and generating submission files for the corresponding EGO4D challenges.

## Requirements
The code is tested to work correctly with:

- GPU environment
- Anaconda Python 3.6.4
- [Pytorch](https://pytorch.org/) v1.2.0
- NumPy
- OpenCV
- [tqdm](https://github.com/tqdm/tqdm)
- [PyAV](https://pypi.org/project/av/)


### Running the code
To run the PNR localization experiment, after preparing the data, run
```
python main.py --task PNR --mode trainval (--other dataset specific arguments)
```
```
python main.py --task PNR --mode test (--other dataset specific arguments)
```

To run the state change classification experiment,
```
python main.py --task State_change --mode trainval (--other dataset specific arguments)
```
```
python main.py --task State_change --mode test (--other dataset specific arguments)
```

other modifiable arguments can be found in `main.py`. For example:
+ --epochs: the maximum training epoch
+ --lr: the learning rate
+ --hidden_size: the hidden size of LSTM, default = 512
+ --num_layers: how many layers in LSTM, default = 1
+ --save_name: the name of the saved best model parameters

For the dataset specific arguments, please refer to the details in other baselines.