"""
Strategy file Lipchitz feature unlearning algorithm
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

class Lipschitz:
    def __init__(
        self,
        model,
        device,
        args,
    ):
        self.model = model
        self.device = device
        self.args = args
        self.opt = torch.optim.SGD(model.parameters(), lr=self.args.lr)

    def lipschitz_optimization(
            self,
            unlearn_trainloader: DataLoader,
            unlearn_pertubbed_trainloader: DataLoader,
    ) -> torch.nn.Module:

        for (x, _, y), x_pertubbed in tqdm(zip(unlearn_trainloader, unlearn_pertubbed_trainloader),
                                           desc= "Lipschitz feature unlearning"):
            self.opt.zero_grad()

            # Normal input image
            image = x.to(self.device)
            output_image = self.model(image)

            total_loss = 0.0
            for xp in x_pertubbed:
                # Pertubbed input image by adding random noise
                image_pertubbed = xp.to(self.device)
                # Disable gradient for the pertubbed image forward
                with torch.no_grad():
                    output_image_pertubbed = self.model(image_pertubbed)

                # Lipschitz loss computation
                loss = self.lipschitz_loss(image= image,
                                           image_pertubbed= image_pertubbed,
                                           output_image= output_image,
                                           output_image_pertubbed= output_image_pertubbed)
                total_loss += loss

            # Average loss over samples
            avg_loss = total_loss / self.args.sample_number

            # Back prop the loss
            avg_loss.backward()
            self.opt.step()

        return self.model

    def lipschitz_loss(
            self,
            image: torch.Tensor,
            image_pertubbed: torch.Tensor,
            output_image: torch.Tensor,
            output_image_pertubbed: torch.Tensor
    ) -> torch.Tensor:

        # Flatten the image for input vector computation
        flat_image, flat_image_pertubbed = image.view(image.size()[0], -1), image_pertubbed.view(
            image_pertubbed.size()[0], -1)

        # Normalize input and output vector
        norm_input = torch.linalg.vector_norm(flat_image - flat_image_pertubbed, dim=1)
        norm_output = torch.linalg.vector_norm(output_image - output_image_pertubbed, dim=1)

        # Compute lipschitz constant
        lipschitz = norm_output / norm_input  # Output divided by input
        lipschitz_cons = (lipschitz.sum()).abs()  # Absolute value for lipschitz constant

        return lipschitz_cons

def lipschitz_unlearning(
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    pertubbed_trainloader: torch.utils.data.DataLoader,
    device: torch.device,
    args: argparse.Namespace
) -> torch.nn.Module:

    feature_removing = Lipschitz(model=model,
                                 device=device,
                                 args= args)

    model_unlearn = feature_removing.lipschitz_optimization(
        unlearn_trainloader=trainloader,
        unlearn_pertubbed_trainloader=pertubbed_trainloader)

    return model_unlearn