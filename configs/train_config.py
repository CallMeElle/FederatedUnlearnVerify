"""
Training configuration
"""
import argparse
"""
Get Args
"""
parser = argparse.ArgumentParser()

# Device
parser.add_argument("-gpu", action="store_true", help="use gpu or not", default= None)

# Model configuration
parser.add_argument("-weight_path", type= str, default= None,
                    help= "baseline model weight path for fine tuning")
parser.add_argument("-hidden_layer_num", type= int,
                    help= "hidden layer number of the linear model for tabular dataset", default= None)
parser.add_argument("-save_model", action='store_true', help= "option to save the trained model", default= None)
parser.add_argument("-checkpoint", type= str, help= "model folder path", default= None)

# Dataset configuration
parser.add_argument("-root", type= str, help= "data folder path", default= "data/")
parser.add_argument("-dataset", type=str, default = "Cifar10", 
                    choices=["MNist", "FMNist","Cifar10", "Cifar20", "Cifar100", "Celeba", "diabetes", "adult"],
                    help="Image: {MNist, FMNist,Cifar10, Cifar20, Cifar100, Celeba}"
                         "Tabular: {diabetes, adult}")

# Backdoor
parser.add_argument("-trigger_size", type=int, help= "backdoor trigger feature square size", default= 5)
parser.add_argument("-trigger_label", type= int, help= "backdoor trigger label", default= 0)

# Sensitive or Bias
parser.add_argument("-bias_ratio", type= float, help= "bias ratio on the dataset")
parser.add_argument("-mnist_mode", type=str, choices=["digit", "background"],
                    help= "mnist dataset bias on digit or background option")
parser.add_argument("-celeba_classification", type= int,
                    help= "Classfication task for celeba dataset, gender")
parser.add_argument("-celeba_bias_feature", type=int,
                    help="Feature number according to the list for bias training purpose") # 15= eye, 31= mouth
parser.add_argument("-pertubbed_part", type= str,
                    choices= ["mouth, eye, nose, face, face_except_mouth"],
                    help= "part to be added noise on")

# Fl train mode
parser.add_argument("-train_mode", type= str,
                    choices= ["sensitive", "backdoor", "bias"],
                    help= "FL training mode for unlearning scenarios", default= "backdoor")

# FL client configuration
parser.add_argument('-client_num', type=int, help="number of clients: K", default= 10)
parser.add_argument('-frac', type=float,help='the fraction of clients per round: C', default= 0.4)
parser.add_argument('-unlearn_client_index', type= int, default= 0,
                    help= "index of unlearn client, 0 indicating first client (Assumption: unlearn client= first client)")

# FL training hyperparameter
parser.add_argument("-global_epochs", type=int, help="number of epochs ", default= 200)
parser.add_argument("-local_epochs", type=int, help="the number of local epochs: E", default= 5)
parser.add_argument("-batch_size", type= int, help= "training batch size", default= 128)
parser.add_argument("-lr", type=float, help='learning rate', default= 0.0001)
parser.add_argument('-momentum', type=float, help='SGD momentum (default: 0.5)', default= 0.5)
parser.add_argument('-optimizer', type=str, choices= ["sgd", "adam"], help="type of optimizer", default= "sgd")

# Training performance configuration
parser.add_argument('-report_training', action='store_true', help= "option to show training performance")
parser.add_argument('-report_interval', type= int, default= 5, help= "training performance report interval")

parser.add_argument("-seed", type=int, help="seed for runs", default= 0)
arguments = parser.parse_args()