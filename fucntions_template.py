def fit(
    model: torch.nn.Module, 
    train_dataloader: DataLoader, 
    validation_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: callable,
    epochs: int,
    lr_scheduler = None,
    metric_fn = None, 
    device: str = "cpu",
    profiler = None
):
    # List to store the loss and metric score for each epoch
    training_loss_history = []
    loss_history = []
    metric_history = []
    

    # Format the batch status 
    # (sc: current step, st: total steps, l: loss)
    batch_status = lambda sc, st, l: f"\rStep {sc + 1:4} of {st:4}; Average epoch loss: {(l / (sc + 1)):6.5f}    "
    # Format the epoch status 
    # (rc: current epoch, et: total epochs)
    epoch_status = lambda ec, et: f"========== Epoch {ec + 1} of {et} =========="


    # Get the total number of batches
    training_batches = len(train_dataloader)
    if validation_dataloader != None:
        validation_batches = len(validation_dataloader)

    # If an metric funtion in not provide default to this lambda
    if metric_fn == None:
        metric_fn = lambda pred, target: 0


    # Define every how many batches the status need to be updated
    batch_update_f = 1


    # Move the model to the device before training
    model.to(device)

    # Train the model for the defined number of epochs
    for epoch in range(0, epochs):
        print(epoch_status(epoch, epochs))

        # If a scheduler was provided print the learning rate at each epoch
        if lr_scheduler != None:
            my_lr = lr_scheduler.get_last_lr()
            print(f"Training with a lr of {my_lr[0]}")

        # Otherwise just notify about the start of the training
        else:
            print("Training...")


        # Set the model to training mode
        model.train()

        # Used to calculate the average loss of the current epoch
        avg_loss = 0

        # Train the model for all the batch
        for step, batch in enumerate(train_dataloader):
            # Move the batch to the device
            batch = list(map(lambda x: x.to(device), batch))

            # Forward pass
            output = model(
                batch[0]
            )

            # Get loss
            loss = loss_fn(batch[1], output)

            # Backward pass and update avg_loss
            loss.backward()
            avg_loss += loss.item()

            # Update the parameters and clear the gradients
            optimizer.step()
            optimizer.zero_grad()

            # Update the displayed status
            if step % batch_update_f == 0:
                print(batch_status(step, training_batches, avg_loss), end="")


            # If a performance profiler is given update it
            if profiler != None:
                profiler.step()


        # Update the training status one last time
        print(batch_status(training_batches - 1, training_batches, avg_loss))

        # Append avg loss to its history list
        training_loss_history.append(avg_loss / training_batches)



        # Run validation at every epoch if the validation data loader is given
        if validation_dataloader != None:
            print("Validation...")

            # Set the model to evaluation mode
            model.eval()

            # Store the average metric and loss for this epoch
            avg_val_metric, avg_val_loss = 0, 0
            
            for step, batch in enumerate(validation_dataloader):
                # Move the batch to the device
                batch = list(map(lambda x: x.to(device), batch))

                # Forward pass with inference mode
                with torch.inference_mode():
                    output = model(
                        batch[0], 
                        token_type_ids =    None, 
                        attention_mask =    batch[1],
                        labels =            batch[2]    
                    )

                # Get loss
                loss = loss_fn(batch[1], output)

                # Update the avg metric and loss
                avg_val_metric += metric_fn(output, batch[2])
                avg_val_loss += loss.item()

            
                # Update the displayed status
                if step % batch_update_f == 0:
                    print(batch_status(step, validation_batches, avg_val_loss), end="")              


            # Update the training status one last time and print validation result
            print(batch_status(validation_batches - 1, validation_batches, avg_val_loss))
            print(f"Validation result => Loss: {avg_val_loss / validation_batches:6.5f};", end = "")
            print(f"  Metric: {avg_val_metric / validation_batches:6.5f}\n")


            # Append avg loss and metric to their history list
            loss_history.append(avg_val_loss / validation_batches)
            metric_history.append(avg_val_metric / validation_batches)



        # Update the learning rate if a sheduler was given
        if lr_scheduler != None:
            lr_scheduler.step()


    # Return the loss and accuracy history
    return training_loss_history, loss_history, metric_history



# Use the model to make predictions
def predict(
    model: torch.nn.Module, 
    input_batch: list, 
    preprocessing_fn: callable, 
    postprocessing_fn: callable, 
    device: str
):
    # Preproccess the batch
    batch = preprocessing_fn(input_batch)

    # Move the batch to the device
    batch = list(map(lambda x: x.to(device), batch))

    # Forward pass with inference mode
    with torch.inference_mode():
        output = model(
            batch[0]
        )

    # Return the processed output
    return postprocessing_fn(output[0])
