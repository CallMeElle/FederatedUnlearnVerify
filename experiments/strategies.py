from typing import Tuple
from unlearn_strategies import strategies
from experiments import utils
from datasets import metrics
import torch
from pytorch_grad_cam import GradCAM
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Experiments:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def eval(self) -> None:
        unlearn = strategies.FeatureUnlearning(args= self.args)

        # Dataset preparation
        retain_client_trainloader, unlearn_client_trainloader, unlearn_client_pertubbed_trainloader, testloader, model_unlearn = unlearn.load_data()

        # Evaluation
        unlearn_client_acc, retain_client_acc, test_acc = metrics.metrics_unlearn(
            unlearn_client_trainloader=unlearn_client_trainloader,
            retain_client_trainloader=retain_client_trainloader,
            testloader=testloader,
            model_unlearn=model_unlearn,
            device= unlearn.device)

        print(f"Retain Client Accuracy: {retain_client_acc}")
        print(f"Unlearn Client Accuracy: {unlearn_client_acc}")
        print(f"Test Accuracy: {test_acc}")

    def gradcam(self) -> None:

        if self.args.unlearning_scenario not in ["backdoor", "bias"]:
            raise Exception("Select only backdoor or bias unlearning scenario to visualize GradCAM result")

        unlearn = strategies.FeatureUnlearning(args=self.args)

        # Dataset preparation
        retain_client_trainloader, unlearn_client_trainloader, unlearn_client_pertubbed_trainloader, testloader, model_unlearn = unlearn.load_data()

        # Initialise GradCAM
        GradCam = GradCAM(model_unlearn, target_layers= model_unlearn.conv5_x)

        # Initialise gradcam directory
        gradcam_dir = f"result/{self.args.dataset}/{self.args.unlearning_scenario}/"
        utils.create_directory_if_not_exists(file_path= gradcam_dir)

        for batch_idx, (x, _, y) in enumerate(unlearn_client_trainloader):

            images = x.to(unlearn.device)
            # Generate original batch heatmaps images
            heatmaps = torch.from_numpy(GradCam(images))

            for img_idx, (img, htmp) in enumerate(zip(images, heatmaps)):
                # Process heatmap image, higher alpah higher heatmap intensity
                heatmap_img = utils.generate_heatmap(heatmap= htmp, image= img, alpha= 0.5)
                # Save generated heatmap image into saving directory
                utils.save_heatmap(gradcam_dir= gradcam_dir,
                                   heatmap_img= heatmap_img,
                                   batch_idx= batch_idx,
                                   img_idx= img_idx,
                                   batch_size= self.args.batch_size)