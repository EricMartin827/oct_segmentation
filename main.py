################################################################################
################################ Python Imports ################################
################################################################################

import sys
import cv2
import numpy as np
import albumentations as A
import monai
import torch
import warnings
from torchinfo import summary
warnings.filterwarnings("ignore")

################################################################################
################################# Local Imports ################################
################################################################################

from staging_utils import (
    build_wandb_monitor,
    create_path_to_new_weight_file,
    find_path_to_weight_file,
    parse_user_inputs,
    build_data_loaders,
    build_loss_function,
    build_optimizer,
    build_scheduler,
    build_model,
    build_train_evaluator,
    build_test_evaluators
)

from training import (
    train
)

################################################################################
################################# Main Function ################################
################################################################################

def describe_system_settings(device: str):

    print(f'\nRunning On A: {device}\n')

    print(f'Python Version: {sys.version}\n')
    print(f'Numpy Version: {np.__version__}')
    print(f'Torch Version: {torch.__version__}')
    print(f'Albumentations Version:  {A.__version__}')
    print(f'Open CV Version:  {cv2.__version__}')
    print(f'Monai Version: {monai.__version__}\n')


def describe_model(model, loader):
    iterator = iter(loader)
    inputs, _ = next(iterator)
    summary(model, input_size=inputs.shape)

def main():

    args = parse_user_inputs()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and \
        not args.use_cpu else "cpu")

    describe_system_settings(device)

    training, validation, testing = build_data_loaders(args)

    train_set_size, train_loader = training
    val_set_size, val_loader = validation
    _, test_loader   = testing

    if args.train:

        project_name = args.wandb_project_name
        print(f"Training {project_name}. This May Take A While...")

        monitor, config = build_wandb_monitor(args)
        model = build_model(args, config=config, device=device)
        new_weight_file = create_path_to_new_weight_file(args)
        loss_function = build_loss_function(args, config)
        optimizer = build_optimizer(args, model, config)
        scheduler = build_scheduler(args, optimizer)
        
        evaluator = build_train_evaluator(args, device)

        describe_model(model, train_loader)

        train(
            model=model,
            train_set_size=train_set_size,
            train_loader=train_loader,
            val_set_size=val_set_size,
            val_loader=val_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            monitor=monitor,
            max_epochs=config.epochs,
            evaluator=evaluator,
            best_model_weight_file=new_weight_file,
            device=device
            )

    else:

        print(f"Running In Evaluation Mode.")

        model = build_model(args, device=device)

        tuned_weight_file = find_path_to_weight_file(args)
        model.load_state_dict(torch.load(tuned_weight_file))

        val_eval, test_eval = build_test_evaluators(args, device)

        val_eval(model, val_loader)
        test_eval(model, test_loader)

if __name__ == "__main__":
    print("Beginning Exectution. Lets Hope This Works!!!")
    main()
    print("Completed Execution. Have a Nice Day!!!")
