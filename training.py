################################################################################
################################ Python Imports ################################
################################################################################

import pandas as pd

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

################################################################################
################################# Local Imports ################################
################################################################################

################################################################################
################################ Main Functions ################################
################################################################################

def train(model,
          train_loader,
          train_set_size,
          val_loader,
          val_set_size,
          loss_function,
          optimizer,
          scheduler,
          max_epochs,
          evaluator,
          best_model_weight_file,
          device
          ):

    '''
    start a typical PyTorch training
    '''

    ### Do This First Before You Start Optimizing Anything
    try:
        with open(best_model_weight_file, 'w') as f:
            print(f"{best_model_weight_file} Can Be Created")
    except FileNotFoundError:
        raise ValueError(f"Cannot Create {best_model_weight_file}")

    ### Calulate The Number Of Batches Per Epoch (For tracking progress)

    training_batches_per_epoch = train_set_size // train_loader.batch_size
    if (train_set_size % train_loader.batch_size) != 0:
        training_batches_per_epoch += 1

    validation_batches_per_epoch = val_set_size // val_loader.batch_size
    if (val_set_size % train_loader.batch_size) != 0:
        validation_batches_per_epoch += 1

    ### Track the history of metric values
    best_metric = -float('inf')
    best_metric_epoch = 0
    metric_values = [best_metric]

    ### Let Indexing Start At 1 For Loss Tracking
    training_loss_values = [None]
    validation_loss_values = [None]

    lrs = list()

    writer = SummaryWriter()

    print("====Training====")

    for epoch in range(1, max_epochs + 1):

        ### Pretty Printing Is Always Nice :)

        print("-" * 10)
        print(f"epoch {epoch}/{max_epochs}")

        ### Put the model in a trainable mode so that features like dropout
        ### are active. More info ...
        ### https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch

        model.train()
        training_loss, step = 0, 0
        for inputs, labels in train_loader:

            ### Stage the batch to proper device. No copy is performed
            ### if the currect device of inputs and labels match current
            ### system settings.
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()                 # zero out gradients
            outputs = model(inputs)               # forward pass
            loss = loss_function(outputs, labels) # compute loss
            loss.backward()                       # compute gradients
            optimizer.step()                      # update model weights

            step += 1 ### Don't lose track :) 

            lrs.append(optimizer.param_groups[0]["lr"]) # help optimization

            training_loss += loss.item() ### Accumlate The Loss For Epoch

            ### Report and Record This Training Step
            temp = training_batches_per_epoch
            print(f"{step}/{temp}, train_loss: {loss.item():.4f}")
            step_id = (training_batches_per_epoch * epoch) + step
            writer.add_scalar("train_loss", loss.item(), step_id)
        
        # Average the loss across number of batches (training steps)
        training_loss /= step
        training_loss_values.append(training_loss)
        print(f"epoch {epoch} average training loss: {training_loss:.4f}")
        
        # Start Evaluation by putting modle in evaluation mode
        model.eval()
        validation_loss, step = 0, 0
        for inputs, labels in val_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)               # Forward pass
            loss = loss_function(outputs, labels) # Compute loss

            validation_loss += loss.item()

            # Note that step should be called after validate()
            scheduler.step(loss.item())
            step += 1

            ### Report and Record This Validation Step
            temp = validation_batches_per_epoch
            print(f"{step}/{temp}, val_loss: {loss.item():.4f}")
            step_id = (validation_batches_per_epoch * epoch) + step
            writer.add_scalar("val_loss", loss.item(), step_id)

        # Average the loss across number of batches
        validation_loss /= step 
        validation_loss_values.append(validation_loss)
        print(f"epoch {epoch} average validation loss: {validation_loss:.4f}")

        # At specific intervals, compute dice metric (expensive operation)
        if (evaluator.should_run_at(epoch)):

            print("-" * 20)
            print(f"Running Metric Evaluation For Epoch {epoch}")

            metric = evaluator(model, val_loader)

            print(f"Evaluation Metric Score {metric} @ Epoch {epoch}")
            print("-" * 20)
            
            metric_values.append(metric)
            
            if metric > best_metric:
        
                print("-" * 20)

                print(f"Saving A New Best Metric Model @ Epoch {epoch}")
                print(f"New Best Mean Dice {metric} For Validation Dataset.")
                print(f"Old Score {best_metric} @ Epoch {best_metric_epoch}")

                best_metric, best_metric_epoch = metric, epoch
                torch.save(model.state_dict(), best_model_weight_file)

                print(f"Weight File Saved @ {best_model_weight_file}")
                writer.add_scalar("val_mean_dice", metric, epoch)

                print("-" * 20)

    print("-" * 10)
    print("--Traning Complete--") 
    print("-" * 10)

    print(f"Best Class 2 Dice Score {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
