"""
Unlearn configuration
"""
import argparse

"""
Get Args
"""
parser = argparse.ArgumentParser()

# Device
parser.add_argument("-gpu", action="store_true", help="use gpu or not")

# Model configuration
parser.add_argument("-weight_path", type=str,
                    default= None,
                    help="Path to pretrained model weights")
parser.add_argument("-hidden_layer_num", type= int,
                    help= "hidden layer number of the linear model for tabular dataset")
parser.add_argument("-save_model", action='store_true', help= "option to save the unlearned model")
parser.add_argument("-checkpoint", type= str, help= "model folder path")

# Dataset configuration
parser.add_argument("-root", type=str, help= "data folder path")
parser.add_argument("-preprocessed_dir", type= str, default= "preprocessed_data")
parser.add_argument("-dataset", type=str, default= "Cifar10",
                    choices=["MNist", "FMNist","Cifar10", "Cifar20", "Cifar100", "Celeba", "adult", "diabetes"],
                    help="dataset for feature unlearning")
parser.add_argument("-save_preprocessed", action= "store_true", help= "option to save the preprocessed data into pkl")

# FL client configuration
parser.add_argument('-client_num', type=int, help="number of clients: K")
parser.add_argument('-frac', type=float, default= 0.4, help='the fraction of clients per round: C')
parser.add_argument('-unlearn_client_index', type= int, default= 0,help= "index of unlearn client, 0 indicating first client (Assumption: unlearn client= first client)")

# Sensitive
parser.add_argument("-reduced", action='store_true',
                    help= "Option to reduced the number of dataset")
parser.add_argument("-celeba_classification", type= int,
                    help= "Classfication task for celeba dataset, gender")
parser.add_argument("-celeba_bias_feature", type=int,
                    help="Feature number according to the list for bias training purpose") # 15= eye, 31= mouth
parser.add_argument("-pertubbed_part", type= str,
                    choices= ["mouth", "eye", "nose", "face", "face_except_mouth"],
                    help= "part to be added noise on")

# Backdoor
parser.add_argument("-trigger_size", type=int, help= "backdoor trigger feature square size")
parser.add_argument("-trigger_label", type= int, help= "backdoor trigger label")

# Bias
parser.add_argument("-mnist_mode", type=str, choices=["digit", "background"])
parser.add_argument("-bias_ratio", type= float, help= "bias ratio on the dataset")

# Unlearn hyperprarameter
parser.add_argument("-n_epochs", type=int, default= 1, help="number of epochs of unlearning method to use")
parser.add_argument("-lambda_coef", type= float, default= 1.0, help= "lambda coefficient to control the computed lipschitz constant")
parser.add_argument("-sigma", type= float, help= "Gaussian noise sigma")
parser.add_argument("-sample_number", type=int, help= "Noise sampling number")
parser.add_argument("-min_sigma", type=float, help= "Maximum sigma noise")
parser.add_argument("-max_sigma", type=float, help= "Minimum sigma noise")
parser.add_argument("-batch_size", type=int, help="batch size for dataloader")
parser.add_argument("-lr", type=float, help="initial learning rate")

# Unlearn configuration
parser.add_argument("-unlearn", action= "store_false", help= "unlearn option")
parser.add_argument("-unlearning_scenario", type= str, choices= ["sensitive", "backdoor", "bias"],
                    help= "unlearning scenarios configuration")
parser.add_argument("-lipschitz_mode", type=str, default="multiple", choices=["single", "multiple"],
                    help= "single noise sampling or multiple noise sampling on unlearn sample")

# Seed
parser.add_argument("-seed", type=int, help="seed for runs")
arguments = parser.parse_args()