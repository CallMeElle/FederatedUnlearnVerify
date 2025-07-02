"""
Tabular dataset
"""
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def diabetes(test_size= 0.1, mode= "train", unlearn_feature= 0, root= "./data"):
    """
    Diabetes dataset: https://www.kaggle.com/datasets/mathchi/diabetes-data-set
    :param test_size: train and test split ratio
    :param mode: train, unlearn or retrain
    :param unlearn_feature: column of feature within X to be removed in int, 0= pregnancies
    :return: trainset: a list [train feature, _, label]
    :return: testset: a list [test feature, _, label]
    :return: perturbed_trainset: a list [perturbed train feature, _, label]
    :return: perturbed_testset: a list [perturbed test feature, _, label]
    """

    if mode not in ['train', 'retrain', 'unlearn', 'mia']:
        raise Exception("Enter correct mode")

    def encode_pregnancies(pregnancies):
        if pregnancies == 0: # not pregnant before
            return 0
        else:
            return 1

    def encode_pregnancies_invert(pregnancies):
        if pregnancies == 0: # not pregnant before
            return 1
        else:
            return 0

    dataset_path = f'{root}/diabetes.csv'
    df = pd.read_csv(dataset_path)

    # Convert 'Pregnancies' column
    if mode in ['train', 'retrain', 'unlearn']:
        df['Pregnancies'] = df['Pregnancies'].apply(encode_pregnancies)  # pregnant before= 1, no pregnant before= 0
    else:
        invert_x = df.drop('Outcome', axis=1) # invert the unlearn feature column for target model set
        invert_x['Pregnancies'] = invert_x['Pregnancies'].apply(encode_pregnancies_invert)
        invert_y = df['Outcome']

        mia_x = df.drop(['Pregnancies'], axis=1)  # not include unlearn feature for mia
        mia_y = df['Pregnancies'].apply(encode_pregnancies_invert)  # unlearn feature as label for mia mode

        x_mia_gtlabel = df['Pregnancies'].apply(encode_pregnancies) # groudtruth label for the unlearn feature

    X = df.drop('Outcome', axis=1)  # independent Feature
    y = df['Outcome']  # dependent Feature
    features = X.columns.tolist()

    if mode == "retrain":
        X = df.drop(features[unlearn_feature], axis= 1)

    if mode in ['train', 'retrain']:
        trainset = []
        testset = []

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        input_train = torch.FloatTensor(X_train.values)
        input_test = torch.FloatTensor(X_test.values)
        label_train = torch.LongTensor(y_train.values)
        label_test = torch.LongTensor(y_test.values)

        for x_train, y_train in zip(input_train, label_train):
            trainset.append([x_train, torch.Tensor([]), y_train])

        for x_test, y_test in zip(input_test, label_test):
            testset.append([x_test, torch.Tensor([]), y_test])

        return trainset, testset

    elif mode == 'mia':
        invert_trainset = []
        invert_testset = []
        mia_trainset = []
        mia_testset = []

        mia_trainlabel_gt = []
        mia_testlabel_gt = []

        # target model initial trainset - random initial unlearn feature value
        invert_X_train, invert_X_test, invert_y_train, invert_y_test = train_test_split(invert_x,
                                                                                        invert_y,
                                                                                        test_size=test_size,
                                                                                        random_state=0)

        invert_input_train = torch.FloatTensor(invert_X_train.values)
        invert_input_test = torch.FloatTensor(invert_X_test.values)
        invert_label_train = torch.LongTensor(invert_y_train.values)
        invert_label_test = torch.LongTensor(invert_y_test.values)

        for x_train, y_train in zip(invert_input_train, invert_label_train):
            invert_trainset.append([x_train, torch.Tensor([]), y_train])

        for x_test, y_test in zip(invert_input_test, invert_label_test):
            invert_testset.append([x_test, torch.Tensor([]), y_test])

        # mia dataset: unlearn feature as label
        mia_X_train, mia_X_test, mia_y_train, mia_y_test = train_test_split(mia_x,
                                                                            mia_y,
                                                                            test_size=test_size,
                                                                            random_state=0)
        mia_input_train = torch.FloatTensor(mia_X_train.values)
        mia_input_test = torch.FloatTensor(mia_X_test.values)
        mia_label_train = torch.LongTensor(mia_y_train.values)
        mia_label_test = torch.LongTensor(mia_y_test.values)

        for x_train, y_train in zip(mia_input_train, mia_label_train):
            mia_trainset.append([x_train, torch.Tensor([]), y_train])

        for x_test, y_test in zip(mia_input_test, mia_label_test):
            mia_testset.append([x_test, torch.Tensor([]), y_test])

        # mia groundtruth label: groundtruth label of the unlearn feature
        x_mia_gtlabel_train, x_mia_gtlabel_test = train_test_split(x_mia_gtlabel,
                                                                   test_size=test_size,
                                                                   random_state=0)
        mia_gtlabel_train = torch.LongTensor(x_mia_gtlabel_train.values)
        mia_gtlabel_test = torch.LongTensor(x_mia_gtlabel_test.values)

        for label in mia_gtlabel_train:
            mia_trainlabel_gt.append(label)
        for label in mia_gtlabel_test:
            mia_testlabel_gt.append(label)

        return invert_trainset, invert_testset, mia_trainset, mia_testset, mia_trainlabel_gt, mia_testlabel_gt

    else:
        raise Exception("Enter correct unlearn mode")

def adult(test_size= 0.1, mode= "train", unlearn_feature= 4, root= "./data"):
    """
    Adule census income dataset: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
    :param test_size: train and test split ratio
    :param mode: train, unlearn or retrain
    :param unlearn_feature: column of feature within X to be removed in int, 4= marital status
    :return: trainset: a list [train feature, _, label]
    :return: testset: a list [test feature, _, label]
    :return: perturbed_trainset: a list [perturbed train feature, _, label]
    :return: perturbed_testset: a list [perturbed test feature, _, label]
    """

    if mode not in ['train', 'retrain', 'unlearn', 'mia']:
        raise Exception("Enter correct mode")

    def encode_marital_status(status):
        if status in ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']:
            return 1
        else:
            return 0

    def encode_marital_status_invert(status):
        if status == 1:
            return 0
        else:
            return 1

    # Load dataset csv path
    dataset_path = f'{root}/adult.csv'
    df = pd.read_csv(dataset_path)

    # Remove 'fnlwgt' column
    df.drop('fnlwgt', axis=1, inplace=True)

    # Convert 'marital-status' column
    df['marital-status'] = df['marital-status'].apply(encode_marital_status) # married= 1, single= 0

    # Convert categorical variables to numerical using label encoding
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Split features (X) and target (y)
    X = df.drop('income', axis=1)
    y = df['income']
    features = X.columns.tolist()

    if mode == 'mia':
        invert_x = df.drop('income', axis=1)  # invert the unlearn feature column for target model set
        invert_x['marital-status'] = invert_x['marital-status'].apply(encode_marital_status_invert)
        invert_y = df['income']

        mia_x = df.drop(['marital-status'], axis=1)  # not include unlearn feature for mia
        mia_y = df['marital-status'].apply(encode_marital_status_invert)  # inverted unlearn feature as label for initial target model trainset

        x_mia_gtlabel = df['marital-status'] # groudtruth label for the unlearn feature

    if mode == "retrain":
        # X = df.drop(features[unlearn_feature], axis= 1)
        feature_to_drop = X.columns[unlearn_feature]
        X = X.drop(feature_to_drop, axis=1)

    if mode in ['train', 'retrain']:
        trainset = []
        testset = []

        # Split train and test dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        input_train = torch.FloatTensor(X_train.values)
        input_test = torch.FloatTensor(X_test.values)
        label_train = torch.LongTensor(y_train.values)
        label_test = torch.LongTensor(y_test.values)

        for x_train, y_train in zip(input_train, label_train):
            trainset.append([x_train, torch.Tensor([]), y_train])

        for x_test, y_test in zip(input_test, label_test):
            testset.append([x_test, torch.Tensor([]), y_test])

        return trainset, testset

    elif mode == 'mia':
        invert_trainset = []
        invert_testset = []
        mia_trainset = []
        mia_testset = []
        mia_trainlabel_gt = []
        mia_testlabel_gt = []

        # target model initial trainset - random initial unlearn feature value
        invert_X_train, invert_X_test, invert_y_train, invert_y_test = train_test_split(invert_x,
                                                                                        invert_y,
                                                                                        test_size=test_size,
                                                                                        random_state=0)

        invert_input_train = torch.FloatTensor(invert_X_train.values)
        invert_input_test = torch.FloatTensor(invert_X_test.values)
        invert_label_train = torch.LongTensor(invert_y_train.values)
        invert_label_test = torch.LongTensor(invert_y_test.values)

        for x_train, y_train in zip(invert_input_train, invert_label_train):
            invert_trainset.append([x_train, torch.Tensor([]), y_train])

        for x_test, y_test in zip(invert_input_test, invert_label_test):
            invert_testset.append([x_test, torch.Tensor([]), y_test])

        # mia trainset - unlearn feature as label
        mia_X_train, mia_X_test, mia_y_train, mia_y_test = train_test_split(mia_x,
                                                                            mia_y,
                                                                            test_size=test_size,
                                                                            random_state=0)
        mia_input_train = torch.FloatTensor(mia_X_train.values)
        mia_input_test = torch.FloatTensor(mia_X_test.values)
        mia_label_train = torch.LongTensor(mia_y_train.values)
        mia_label_test = torch.LongTensor(mia_y_test.values)

        for x_train, y_train in zip(mia_input_train, mia_label_train):
            mia_trainset.append([x_train, torch.Tensor([]), y_train])

        for x_test, y_test in zip(mia_input_test, mia_label_test):
            mia_testset.append([x_test, torch.Tensor([]), y_test])

        # mia groundtruth label
        x_mia_gtlabel_train, x_mia_gtlabel_test = train_test_split(x_mia_gtlabel,
                                                                   test_size=test_size,
                                                                   random_state=0)
        mia_gtlabel_train = torch.LongTensor(x_mia_gtlabel_train.values)
        mia_gtlabel_test = torch.LongTensor(x_mia_gtlabel_test.values)

        for label in mia_gtlabel_train:
            mia_trainlabel_gt.append(label)
        for label in mia_gtlabel_test:
            mia_testlabel_gt.append(label)

        return invert_trainset, invert_testset, mia_trainset, mia_testset, mia_trainlabel_gt, mia_testlabel_gt

    else:
        raise Exception("Enter correct mode")

def gss(test_size= 0.1, mode= "train", unlearn_feature= 9, root= "./data"):
    """
    :param test_size: train and test split ratio
    :param mode: train, unlearn or retrain
    :param unlearn_feature: column of feature within X to be removed in int, 4= marital status
    :return: trainset: a list [train feature, _, label]
    :return: testset: a list [test feature, _, label]
    :return: perturbed_trainset: a list [perturbed train feature, _, label]
    :return: perturbed_testset: a list [perturbed test feature, _, label]
    """

    if mode not in ['train', 'retrain', 'unlearn']:
        raise Exception("Enter correct mode")

    def encode_pornlaw(pornlaw):
        if pornlaw == 'legal':
            return 0

        elif pornlaw in ['illegal', 'illegalillegal8 ']:
            return 1

        else:
            return 2

    def encode_divorce(divorce):
        if divorce in ['ds_yes', 'noanswer']:
            return 1
        else:
            return 0

    def encode_relig(relig):
        if relig == 'protestant':
            return 0
        elif relig == 'catholic':
            return 1
        else:
            return 2

    # Load dataset csv path
    dataset_path = f'{root}/GSS.csv'
    df = pd.read_csv(dataset_path)

    # Filter rows where marital status is not "separated", since seperated only 1, may cause overfitting
    df = df[df['marital'] != 'separated']

    # Convert 'pornlaw' column
    df['pornlaw'] = df['pornlaw'].apply(encode_pornlaw)  # legal= 0, illegal= 1

    # Convert 'divorce' column
    df['divorce'] = df['divorce'].apply(encode_divorce) # ds_yes= 1, ds_no= 0

    # Convert 'relig' column
    df['relig'] = df['relig'].apply(encode_relig) # protestant= 0, catholic= 1, other= 2

    # Convert categorical variables to numerical using label encoding
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Split features (X) and target (y)
    X = df.drop('hapmar', axis=1)
    y = df['hapmar']
    features = X.columns.tolist()

    if mode == "retrain":
        #X = df.drop(features[unlearn_feature], axis= 1)
        feature_to_drop = X.columns[unlearn_feature]
        X = X.drop(feature_to_drop, axis=1)

    if mode in ['train', 'retrain']:
        trainset = []
        testset = []

        # Split train and test dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        input_train = torch.FloatTensor(X_train.values)
        input_test = torch.FloatTensor(X_test.values)
        label_train = torch.LongTensor(y_train.values)
        label_test = torch.LongTensor(y_test.values)

        for x_train, y_train in zip(input_train,label_train):
            trainset.append([x_train, torch.Tensor([]), y_train])

        for x_test, y_test in zip(input_test, label_test):
            testset.append([x_test, torch.Tensor([]), y_test])

        return trainset, testset

    # Unlearn mode
    elif mode == 'unlearn':

        trainset = []
        testset = []
        perturbed_trainset = []
        perturbed_testset = []

        # Copy to make it mutable
        x_perturbed = X.values.copy()

        # Compute mean and standard deviation for the targeted feature
        mean = df[features[unlearn_feature]].mean()
        sigma = df[features[unlearn_feature]].std()

        # Define size of the gaussian noise generated
        noise_size = len(X[features[unlearn_feature]])
        # Generate gaussian noise
        gaussian_noise = np.random.normal(loc=mean, scale=sigma, size=noise_size)

        for feature, noise in zip(x_perturbed, gaussian_noise):
            feature[unlearn_feature] += noise  # Inject gaussian noise into target feature in X

        X_perturbed_train, X_perturbed_test, _, _ = train_test_split(x_perturbed, y, test_size=test_size,
                                                                     random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        input_perturbed_train = torch.FloatTensor(X_perturbed_train)
        input_perturbed_test = torch.FloatTensor(X_perturbed_test)

        input_train = torch.FloatTensor(X_train.values)
        input_test = torch.FloatTensor(X_test.values)
        label_train = torch.LongTensor(y_train.values)
        label_test = torch.LongTensor(y_test.values)

        for x_train, y_train, x_p_train in zip(input_train, label_train, input_perturbed_train):
            trainset.append([x_train, torch.Tensor([]), y_train])
            perturbed_trainset.append([x_p_train, torch.Tensor([]), y_train])

        for x_test, y_test, x_p_test in zip(input_test, label_test, input_perturbed_test):
            testset.append([x_test, torch.Tensor([]), y_test])
            perturbed_testset.append([x_p_test, torch.Tensor([]), y_test])

        return trainset, testset, perturbed_trainset, perturbed_testset

    else:
        raise Exception("Enter correct mode")