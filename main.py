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

warnings.filterwarnings("ignore")

################################################################################
################################# Local Imports ################################
################################################################################

from staging_utils import (
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


def main():

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    describe_system_settings(device)

    args = parse_user_inputs()

    training, validation, testing = build_data_loaders(args)

    train_set_size, train_loader = training
    val_set_size, val_loader     = validation
    _, test_loader   = testing

    model = build_model(args, device)

    if args.train:

        print(f"Running In Training Mode. This May Take A While...")

        new_weight_file = create_path_to_new_weight_file(args)
        loss_function = build_loss_function(args)
        optimizer = build_optimizer(args, model)
        scheduler = build_scheduler(args, optimizer)
        
        evaluator = build_train_evaluator(args, device)

        train(
            model=model,
            train_set_size=train_set_size,
            train_loader=train_loader,
            val_set_size=val_set_size,
            val_loader=val_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epochs=args.epochs,
            evaluator=evaluator,
            best_model_weight_file=new_weight_file,
            device=device
            )
    else:

        print(f"Running In Evaluation Mode.")

        tuned_weight_file = find_path_to_weight_file(args)
        model.load_state_dict(torch.load(tuned_weight_file))

        val_eval, test_eval = build_test_evaluators(args, device)

        val_eval(model, val_loader)
        test_eval(model, test_loader)

if __name__ == "__main__":
    print("Beginning Exectution. Lets Hope This Works!!!")
    main()
    print("Completed Execution. Have a Nice Day!!!")
