# FederatedUnlearnVerify


Changes to original
to start learning create a data folder in the repo that the data can be saved there and define the path during learning as flag.
Additional resources in requirements.txt

## Run code
To start learning use this parameters

`python fl_training_main.py -train_mode backdoor -dataset Cifar10 -root ./data -trigger_label 0 -trigger_size 5 -global_epochs 200 -local_epochs 5 -batch_size 128 -lr 0.0001  -client_num 10 -frac 0.4 -momentum 0.5 -optimizer 'sgd' -seed 0 -report_training -save_model`

## Sources and License
This repository is based on https://github.com/OngWinKent/Federated-Feature-Unlearning which implements Learning, Unlearning and Verification by using feature sensitivity, model inversion attacks and measuring accuracy on the retain dataset. In addition to that I have implemented additional verification methods and have made changes to the original code by Win Kent Ong to make usage easier. 
Just as the original code this project is open source under BSD-3 license.
