
import torch
import torch.optim as optim
import torch.nn as nn
import os
from typing import Literal

from model import Net

class Solver(object):
    """Solver for training and testing."""

    def __init__(self, train_loader, val_loader, device, writer, args):
        """Initialize configurations."""

        self.args = args
        self.model_name = 'model_{}.pth'.format(self.args.model_name)

        # Define the model
        self.net = Net(self.args, train_loader.dataset.num_classes()).to(device)
        print(self.net)
        # Load a pretrained model
        if self.args.resume_train:
            self.load_model()
        
        # Define Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Choose optimizer
        if self.args.opt == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9)
        elif self.args.opt == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)

        # Epochs
        self.epochs = self.args.epochs

        # Early stopping
        self.early_stopping_enable = (self.args.early_stopping != None and self.args.early_stopping >= 0)
        if self.early_stopping_enable:
            self.max_eval_val = 0   # best-so-far model evaluation on validation set
            self.non_imp = 0   # number of non-improvements on validation set
            self.max_non_imp = self.args.early_stopping

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device
        self.device = device

        # Tensorboard writer
        self.writer = writer

    def save_model(self):
        # If you want to save the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        os.makedirs(os.path.dirname(check_path), exist_ok=True) # create dir if it doesn't exist
        torch.save(self.net.state_dict(), check_path)
        print('Model saved!')

    def load_model(self):
        # Function to load the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        self.net.load_state_dict(torch.load(check_path))
        print('Model loaded!')

    def early_stopping(self) -> bool:
        return (self.non_imp > self.max_non_imp) if self.early_stopping_enable else False
    
    def train(self):
        # Store principal information on tensorboard
        self.writer.add_text(
            'Info and Settings',
            f'Run name: {self.args.run_name}\n' +
            f'Model name: {self.args.model_name}\n' +
            '\n' +
            f'Model: ResNet-{self.args.depth}\n' +
            f'Pretrained: {'Yes' if self.args.pretrained else 'No'}\n' +
            '\n' +
            f'Optimizer: {self.args.opt}\n' +
            f'Epochs: {self.args.epochs}\n' +
            f'Batch Size: {self.args.batch_size}\n' +
            f'Learning Rate: {self.args.lr}\n' +
            f'Norm layers: {'Yes' if self.args.use_norm else 'No'}\n' +
            f'Early stopping: {f'After {self.args.early_stopping} non-improvements' if self.early_stopping_enable else 'No'}'
        )

        # TRAINING
        self.net.train()
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                
                # Put data on correct device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Print statistics
                running_loss += loss.item()
                if i % self.args.print_every == self.args.print_every - 1: 
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / self.args.print_every:.3f}')

                    self.writer.add_scalar(
                        'Training Loss',
                        running_loss / self.args.print_every,
                        epoch * len(self.train_loader) + i
                    )
                    
                    running_loss = 0.0

                    # Accuracy on training and validation set
                    self.evaluation(epoch, i, subset='train')
                    self.evaluation(epoch, i, subset='val')
                    if self.early_stopping():
                        break
            
            if not self.early_stopping_enable:  # otherwise model is saved for each improvement on validation
                self.save_model()
            if self.early_stopping():
                break
        
        self.writer.flush()
        self.writer.close()
        print('Finished Training' + (' (early stop)' if epoch < self.epochs - 1 else ''))  
    
    def evaluation(self, epoch, i, subset: Literal['train', 'val']='val'):
        # Now lets evaluate the model (on training set or validation one)
        correct = 0
        total = 0

        # Put net into evaluation mode
        self.net.eval()

        # Since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in (self.val_loader if subset == 'val' else self.train_loader):
                inputs, labels = data
                # Put data on correct device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Calculate outputs by running images through the network
                outputs = self.net(inputs)
                # The class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        self.writer.add_scalar(
            ('Validation' if subset == 'val' else 'Training') + 'Accuracy',
            accuracy,
            epoch * len(self.train_loader) + i
        )
        if self.early_stopping_enable and subset == 'val':
            if accuracy > self.max_eval_val: # improvement
                self.max_eval_val = accuracy
                self.non_imp = 0
                self.save_model() # after an improvement, let's save the model
            else:
                self.non_imp += 1

        print(f'Accuracy of the network on the {len((self.val_loader if subset == 'val' else self.train_loader).dataset)} {'validation' if subset == 'val' else 'training'} images: {100 * correct / total} %' + 
              (f' [non-improvements: {self.non_imp}/{self.max_non_imp}]' if self.early_stopping_enable and subset == 'val' else ''))
        self.net.train()
