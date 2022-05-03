################################################################################
################################ Python Imports ################################
################################################################################

import numpy as np

import matplotlib.pyplot as plt

from monai.transforms import (
    AsDiscrete,
)

################################################################################
################################# Local Imports ################################
################################################################################

from oct.aux import (
    predict
)

################################################################################
################################ Main Functions ################################
################################################################################

def visualize(image, mask, output, uncertainty=None,
              index=0, filepath=None):

    n_images = (3, 4)[uncertainty is not None]
    _, axes = plt.subplots(nrows=1, ncols=n_images, figsize=(12, 5))

    [axes[i].axis("off") for i in range(n_images)]
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('OCT Image ' + str(index))

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask ' + str(index))

    axes[2].imshow(output, cmap='gray')
    axes[2].set_title('Output ' + str(index))

    if uncertainty is not None:
        _ = axes[3].imshow(uncertainty, cmap="jet")
        axes[3].set_title('Uncertainty ' + str(index))

    plt.subplots_adjust(wspace=.1, hspace=.05)

    if (filepath):
        plt.savefig(filepath, facecolor='w', transparent=False)
    else:
        plt.show()

    plt.close('all')

def visualize_group(model, loader, K=1, uncertainty=True, device="cuda"):

    model.eval()
    argmax = AsDiscrete(argmax=True, to_onehot=None)
    batch, N = 0, loader.batch_size

    for image, label in loader: ### For Each Batch in Dataset

        image, label = image.to(device), label.to(device)

        output_probs, output_uncertainty = predict(model, image, K)

        batch += 1
        for i in range(N): ### For Each Sample in Batch

            sample_id = (N * (batch - 1)) + (i + 1)
            print(f"Creating Visual For Sample {sample_id}")

            img   = image[i, 0, :, :].cpu().detach().numpy()  # Get 2D Rep of Pixels
            mask  = label[i, 0, :, :].cpu().detach().numpy()  # Get 2D Rep of Labels
            preds = argmax(output_probs[i])[0].cpu().detach().numpy()

            if uncertainty:
                entropy = output_uncertainty[i, :, :].cpu().detach().numpy()
                visualize(img, mask, preds, entropy, sample_id)
            else:
                visualize(img, mask, preds, None, sample_id)

def plot_loss_curves(training_loss, validation_loss,
                     learning_rate_scheduler,
                     loss_curves_image_file=None,
                     visualize_loss=True):

    ### Display The Train and Validation Loss Curves
    _, axs = plt.subplots(2, 1)

    time_steps_training = np.arange(0, len(training_loss))
    time_steps_validation = np.arange(0, len(validation_loss))

    l1, = axs[0].plot(time_steps_training, training_loss)
    l2, = axs[0].plot(time_steps_validation, validation_loss)

    axs[0].legend(
        (l1, l2),
        ('Training Loss', 'Validation Loss'),
        loc='upper right',
        shadow=True)

    axs[0].set_title("Training Validation and losses per iteration")
    axs[0].set_ylabel("loss")
    axs[0].set_xlabel("iter")

    axs[1].plot(np.arange(0, len(learning_rate_scheduler)), learning_rate_scheduler)
    axs[1].set_title("Learning rate rescheduler per iteration")
    axs[1].set_ylabel("loss")
    axs[1].set_xlabel("iter")

    if (loss_curves_image_file is not None):
        plt.savefig(loss_curves_image_file)

    if visualize_loss:
        plt.show()
