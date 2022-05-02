################################################################################
################################ Python Imports ################################
################################################################################

import os
import torch

from monai.data import decollate_batch

from monai.transforms import (
    AsDiscrete,
    Compose
)

from monai.networks.nets import (
    UNet
)
from monai.metrics import (
    compute_meandice,
)

################################################################################
################################# Local Imports ################################
################################################################################

from visual import (
    visualize,
)

from evaluation import (
    display_dice_scores,
)

from aux import (
    predict,
)

################################################################################
############################### Main Functions #################################
################################################################################

class PostEvaluationImageGenertor:

    def __init__(self,
                 k_stochasitc_passes=1,
                 num_classes=3,
                 device='cuda',
                 desc="No Description"
                 ):

        self.K = k_stochasitc_passes
        self.num_classes = num_classes
        self.device = device
        self.desc = desc

        if not os.path.exists(self.output_dir):
            raise ValueError(f"{self.output_dir} Does Not Exist!")

    def __call__(self, model, loader, output_dir):

        if not os.path.exists(self.output_dir):
            raise ValueError(f"{self.output_dir} Does Not Exist!")

        return self.__evaluate(
            model=model,
            loader=loader,
            output_dir=output_dir,
            K=self.K,
            num_classes=self.num_classes,
            device=self.device,
            desc=self.desc)

    def __evaluate(
            self,
            model, loader, output_dir,
            K=1,
            num_classes=3,
            device="cuda",
            desc="None Specified"):

        print(f"==== Evaluation {desc} ====")

        if K > 1: print("Running K stochastic forward passes prediction...")
        else:     print("Running standard pointwise prediction...")

            ### Set Post Processing To Automatically One-Hot Encode Outputs
        post_processing = Compose(
            [AsDiscrete(argmax=True, to_onehot=num_classes)])
        label_processing = Compose(
            [AsDiscrete(argmax=False, to_onehot=num_classes)])

        sample = 0
        argmax = AsDiscrete(argmax=True, to_onehot=None)

        # produce predictions for each batch
        with torch.no_grad():

            for images, labels in loader:

                images, labels = images.to(device), labels.to(device)

                # Generate Prediction Along With Model's confidence in it.
                # predict() applies softmax to raw logits outputed from the model
                output_probs, output_uncertainty = predict(model, images, K)

                preds_1hot = [post_processing(i).unsqueeze(dim=0) \
                              for i in decollate_batch(output_probs)]
                truth_1hot = [label_processing(i).unsqueeze(dim=0) \
                              for i in decollate_batch(labels)]

                #### Generate [N, C, H, W] One Hot Pytorch Tensors To Evaluate
                #### Each element of the above lists are [1, C, H, W]
                preds_1hot = torch.cat(preds_1hot, dim=0)
                truth_1hot = torch.cat(truth_1hot, dim=0)

                ### Creates A List of [batch_size, class_scores] elements.
                dice_metrics.append(
                    compute_meandice(
                        y_pred=preds_1hot, y=truth_1hot))

                ### Generate Predictions and uncertainty across all samples
                ### fot this batch.
                for image, label, prob, uncertainty in \
                    zip(images, labels, output_probs, output_uncertainty):

                    sample += 1
                    print(f"Creating Visual For Instance {desc}-{sample}")

                    filename = str(f"{desc}-{sample}.png")
                    filepath = os.path.join(output_dir, filename)

                    img     = image[0].cpu()
                    mask    = label[0].cpu()
                    preds   = argmax(prob)[0].cpu()
                    entropy = uncertainty[0].cpu()

                    visualize(image=img,
                              mask=mask,
                              output=preds,
                              uncertainty=entropy,
                              index=sample,
                              filepath=filepath)

        ### Collate the gathered metrics for recording summary statistics for CSV
        dice_metrics = torch.cat(dice_metrics, dim=0) ### yields [N, NUM_CLASSES]
        avg_dice = torch.mean(dice_metrics, dim=0) ### Along Sample Dimension
        display_dice_scores(avg_dice)

        return avg_dice[2] ### RNFL Layer Dice Score (Class 2)
