# FederatedUnlearnVerify

## Setup
1. Install python 3.9
2. Create virtual environment
3. Install requirements.txt in the virtual environment with pip
    + bz2 needs to be installed on Linux and removed when installing on Windows
    + `pip install -r requirements.txt`

## Run code
To start learning run 
`python fl_training_main.py` which trains the model with the following parameters:
`python fl_training_main.py -train_mode backdoor -dataset Cifar10 -root ./data -trigger_label 0 -trigger_size 5 -global_epochs 200 -local_epochs 5 -batch_size 128 -lr 0.0001  -client_num 10 -frac 0.4 -momentum 0.5 -optimizer 'sgd' -seed 0 -report_training -save_model`

Unlearning runs using
`python unlearn_main.py`

The models are automatically saved in FederatedUnlearnVerify/None/Cifar10/backdoor/ when learning is started with nohup, which continues the training process in the background
For each parameter variation druing learning new pkl files are created in FederatedUnlearnVerify/img_train/

After creating multiple variations of paramters, when only testing unlearning the model and pkl files need to be swapped out accordingly.


##Verification
The verification model is created with 
`python verify_main.py`


## Sources and License
This repository is based on https://github.com/OngWinKent/Federated-Feature-Unlearning which implements Learning, Unlearning and Verification by using feature sensitivity, model inversion attacks and measuring accuracy on the retain dataset. In addition to that I have implemented additional verification methods and have made changes to the original code by Win Kent Ong to make usage easier. 
Just as the original code this project is open source under BSD-3 license.
