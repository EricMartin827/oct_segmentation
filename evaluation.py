################################################################################
################################ Python Imports ################################
################################################################################

import pandas as pd

import torch
import numpy as np
from monai.data import decollate_batch
from monai.transforms import (
    AsDiscrete,
    Compose
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

from aux import (
    compute_auc,
    compute_jaccard,
    compute_accuracy,
    predict
)

################################################################################
######################### Main Functions and Classes ###########################
################################################################################

def evaluate(model,
             loader,
             K=1,
             number_visualize=1,
             uncertainty=True,
             export_path=None,
             num_classes=3,
             device='cuda'):
    '''
    Compute prediction and loss on test data (also used for validation)

    export_path: must be a .xlsx file
    '''

    print("====Evaluation====")

    if K > 1: print("Running K stochastic forward passes prediction...")
    else: print("Running standard pointwise prediction...")

    ### Set Post Processing To Automatically One-Hot Encode Outputs
    post_processing = Compose(
        [AsDiscrete(argmax=True, to_onehot=num_classes)])
    label_processing = Compose(
        [AsDiscrete(argmax=False, to_onehot=num_classes)])

    # hold file to write to csv
    dice_metrics, auc_metrics, acc_metrics, jac_metrics = [], [], [], []

    if export_path is None: full_metrics = False
    else: full_metrics = True

    # produce predictions for each batch
    with torch.no_grad():

        for image, label in loader:

            image, label = image.to(device), label.to(device)

            # Generate Prediction Along With Model's confidence in it.
            # predict() applies softmax to raw logits outputed from the model
            output_probs, output_uncertainty = predict(model, image, K)

            ### Developer Note:
            ### MONAI is big on post processing and the API really wants
            ### you to seperate batch outputs into a list of individual items.
            ### I suspect there is a way to avoid this if runtime is too slow,
            ### but for now no need to get fancy. Monai is still young library.
            ### Decollate means seperate the batch into individual elements.
            ### Need to add a leading dimension (unsqueeze(dim=0)) so that
            ### we have a list of [1, C, H, W] one hot classifications and
            ### labels which me merge along the sample/instance dimension
            ### in the subsequent concatenation.

            preds_1hot = [post_processing(i).unsqueeze(dim=0) \
                          for i in decollate_batch(output_probs)]
            truth_1hot = [label_processing(i).unsqueeze(dim=0) \
                          for i in decollate_batch(label)]

            #### Generate [N, C, H, W] One Hot Pytorch Tensors To Evaluate
            #### Each element of the above lists are [1, C, H, W]
            preds_1hot = torch.cat(preds_1hot, dim=0)
            truth_1hot = torch.cat(truth_1hot, dim=0)

            ### Creates A List of [batch_size, class_scores] elements.
            dice_metrics.append(
                compute_meandice(
                    y_pred=preds_1hot, y=truth_1hot))

            ### These are scalar values unlike DICE above
            if full_metrics:
                auc_metrics.append(compute_auc(preds_1hot, truth_1hot))
                acc_metrics.append(compute_accuracy(preds_1hot, truth_1hot))
                jac_metrics.append(compute_jaccard(preds_1hot, truth_1hot))


    ### Collate the gathered metrics for recording summary statistics for CSV

    dice_metrics = torch.cat(dice_metrics, dim=0) ### yields [N, NUM_CLASSES]
    if full_metrics:
        auc_metrics = torch.cat(auc_metrics, dim=0)
        acc_metrics = torch.cat(acc_metrics, dim=0)
        jac_metrics = torch.cat(jac_metrics, dim=0)

    avg_dice = torch.mean(dice_metrics, dim=0) ### Along Sample Dimension

    print(f"-" * 20)
    print(f"Average DICE coefficient (class 0) : {avg_dice[0]}")
    print(f"Average DICE coefficient (class 1) : {avg_dice[1]}")
    print(f"Average DICE coefficient (class 2) : {avg_dice[2]}")
    print(f"-" * 20)

    ### Save out to excel if applicable
    if export_path:
        writer_xlsx = pd.ExcelWriter(export_path, engine='xlsxwriter')
        C = label.shape[1]
        for c in range(C):
            result = pd.DataFrame(
                          {'dice': dice_metrics[:,c].cpu(),
                          'auc': auc_metrics[:,c].cpu(),
                          'accuracy': acc_metrics[:,c].cpu(),
                          'jaccard': jac_metrics[:,c].cpu()
                          }).astype("float")
            result.loc['Mean'] = result.mean(axis=0)
            result.loc['Std'] = result.std(axis=0)
            result.index.names = ['patch']
            result.to_excel(writer_xlsx, sheet_name='class'+str(c))
        writer_xlsx.save()
        print('Results saved to ' + export_path)

    ### This is the key metric the model must to learn to segment.
    ### Label 2 -> RFNL Layer which thins in aging populations.
    rfnl_dice_score = avg_dice[2]

    if number_visualize <= 0:
        return rfnl_dice_score

    ### Otherwise, lets see how well we are doing visually.
    ### Visualize A Subset Of The Data To Confirm How Well It's Learning

    last_batch_size = len(preds_1hot)
    if number_visualize > last_batch_size:
        number_visualize = last_batch_size

    print(f"Visualizing {number_visualize} Random Instances of The Last Batch")

    ### No one hot representation needed
    argmax = AsDiscrete(argmax=True, to_onehot=None)
    for i in np.random.randint(0, loader.batch_size, number_visualize):

        print(f"Creating Visual For Instance {i} Of Last Batch")

        img   = image[i, 0, :, :].cpu()     # Get 2D Rep of Source Pixels
        mask  = label[i, 0, :, :].cpu()     # Get 2D Rep of Segmentation Labels
        preds = argmax(output_probs[i])[0].cpu()

        if uncertainty:
            entropy = output_uncertainty[i, :, :].cpu()
            visualize(img, mask, preds, entropy, i) # Pass i for ID
        else:
            visualize(img, mask, preds, None, i)

    return rfnl_dice_score

class Evaluator:

    def __init__(self,
                 interval=1,
                 k_stochastic_passes=1,
                 num_to_visualize=0,
                 uncertainty=False,
                 results_file=None,
                 num_classes=3,
                 device='cuda'):

        self.interval = interval
        self.K = k_stochastic_passes
        self.num_to_visaulize = num_to_visualize
        self.uncertainty = uncertainty
        self.results_file = results_file
        self.num_classes = num_classes
        self.device = device

    def __call__(self, model, loader) -> float:

        return evaluate(
            model=model,
            loader=loader,
            K=self.K,
            number_visualize=self.num_to_visaulize,
            uncertainty=self.uncertainty,
            export_path=self.results_file,
            num_classes=self.num_classes,
            device=self.device
        )

    def should_run_at(self, epoch) -> bool:
        return (epoch % self.interval) == 0
