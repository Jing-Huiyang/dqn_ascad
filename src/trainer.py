import torch
import time
import numpy as np
import os
from torch import nn
from src.net import MLP, CNN, weight_init

def trainer(config, num_epochs, num_sample_pts, dataloaders, dataset_sizes, model_type, classes, device, model_root=None, model_idx=None):

    # Build the model
    if model_type == "mlp":
        model = MLP(config, num_sample_pts, classes).to(device)
    elif model_type == "cnn":
        model = CNN(config, num_sample_pts, classes).to(device)
    weight_init(model, config['kernel_initializer'])
    # Creates the optimizer
    lr = config["lr"]
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    # This is the trainning Loop
    criterion = nn.CrossEntropyLoss()
    # scheduler = scheduler
    start = time.time()
    
    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch +1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:  # ,
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            tk0 = dataloaders[phase]  # tqdm(dataloader[phase])
            for batch in tk0:
                inputs = batch['trace'].to(device)
                labels = batch['sensitive'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print("outputs.shape: ", outputs.shape)

                    _, preds = torch.max(outputs, dim=1)

                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            inputs.detach()
            labels.detach()
            
            # Store losses
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)
                
            # Here we calculate the GE, NTGE and the accuracy over the X_attack traces.
            print('{} Epoch Loss: {:.4f} Epoch Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        model.eval()
        # model.to("cpu")
        # if (epoch + 1) % 10 == 0 and epoch != 0:

    print("Finished Training Model")
    
    # Add losses as attributes to the model
    model.train_losses = train_losses
    model.val_losses = val_losses
    
    # Save training history if model_root and model_idx are provided
    if model_root is not None and model_idx is not None:
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        history_path = os.path.join(model_root, f"model_{model_idx}_history.npy")
        np.save(history_path, history)
    
    return model