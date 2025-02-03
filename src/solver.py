
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

        # Best-so-far global accuracy on validation set
        self.best_accuracy = 0

        # Early stopping
        self.early_stopping_enable = (self.args.early_stopping != None and self.args.early_stopping >= 0)
        if self.early_stopping_enable:
            self.non_imp = 0   # number of non-improvements on validation set
            self.max_non_imp = self.args.early_stopping

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device
        self.device = device

        # Absolute iteration of the last saved model
        self.iter_model_save = None
        self.val_accuracy_c_model_save = None

        # Train labels
        self.train_labels = train_loader.dataset.train_labels()

        # Tensorboard writer
        self.writer = writer

    def absolute_iter(self, epoch, i):
        return epoch * len(self.train_loader) + i

    def save_model(self, epoch, i):
        # If you want to save the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        os.makedirs(os.path.dirname(check_path), exist_ok=True) # create dir if it doesn't exist
        torch.save(self.net.state_dict(), check_path)
        self.iter_model_save = self.absolute_iter(epoch, i)
        print('Model saved!')

    def load_model(self):
        # Function to load the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        self.net.load_state_dict(torch.load(check_path))
        print('Model loaded!')

    # It returns True if the early stopping condition is satisfied, False otherwise
    def early_stopping(self) -> bool:
        return (self.non_imp > self.max_non_imp) if self.early_stopping_enable else False
    
    # It saves on tensorboard the class accuracy histogram for the last saved model
    def save_class_accuracy_histogram(self):
        for label in self.train_labels:
            self.writer.add_histogram(
                f'Model class accuracy (absolute iter {self.iter_model_save})',
                self.val_accuracy_c_model_save[label].item(),
                label
            )
    
    def train(self):
        # Store principal information on tensorboard
        self.writer.add_text(
            'Info and Settings',
            f'Run name: {self.args.run_name}\n' +
            f'Model name: {self.args.model_name}\n' +
            '\n' +
            f'Model: ResNet-{self.args.depth}\n' +
            f'Pretrained: {'yes' if self.args.pretrained else 'no'}\n' +
            '\n' +
            f'Optimizer: {self.args.opt}\n' +
            f'Epochs: {self.args.epochs}\n' +
            f'Batch Size: {self.args.batch_size}\n' +
            f'Learning Rate: {self.args.lr}\n' +
            f'Norm layers: {'yes' if self.args.use_norm else 'no'}\n' +
            f'Early stopping: {f'after {self.args.early_stopping} non-improvements' if self.early_stopping_enable else 'no'}\n' +
            '\n' +
            f'Training classes: {
                ', '.join(map(
                    lambda c_l : f'{c_l[1]} ({self.train_loader.dataset.class_description(c_l[0])})',
                    self.train_loader.dataset.training_classes_indexes()
                )) if self.train_loader.dataset.training_with_subset() else 'all'
            }'
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
                last_absolute_iter = (epoch == self.epochs - 1 and i == len(self.train_loader) - 1) # last absolute iteration
                if (i % self.args.print_every == self.args.print_every - 1) or last_absolute_iter:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i % self.args.print_every + 1):.3f}')

                    # plot new loss point
                    self.writer.add_scalar(
                        'Training Loss',
                        running_loss / (i % self.args.print_every + 1),
                        self.absolute_iter(epoch, i)
                    )
                    
                    running_loss = 0.0

                    # accuracy on training and validation set
                    self.evaluation(epoch, i, subset='train')
                    self.evaluation(epoch, i, subset='val') # with model saving if there's an improvement

                    # early stopping condition to exit from iteration loop
                    if self.early_stopping():
                        break

            # Early stopping to exit from epoch loop
            if self.early_stopping():
                break
        
        # Save class accuracy of the saved model
        if self.args.class_accuracy:
            self.save_class_accuracy_histogram()

        self.writer.flush()
        self.writer.close()

        print('Finished Training' + (' (early stop)' if epoch < self.epochs - 1 else ''))  
    
    def evaluation(self, epoch, i, subset: Literal['train', 'val']='val'):
        # Put net into evaluation mode
        self.net.eval()

        # Log using much
        subset_log = ('Validation' if subset == 'val' else 'Training')

        # Select correct loader
        loader = self.val_loader if subset == 'val' else self.train_loader

        # Global accuracy and, eventually, the accuracy for each class
        if self.args.class_accuracy:
            num_classes = loader.dataset.num_classes()
            correct_c = torch.zeros(num_classes).to(self.device)
            total_c = torch.zeros(num_classes).to(self.device)
        else:
            correct = 0
            total = 0

        # Since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in loader:
                inputs, labels = data

                # put data on correct device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # calculate outputs by running images through the network
                outputs = self.net(inputs)

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)

                # correct predictions counting
                if self.args.class_accuracy:
                    correct_mask = predicted == labels
                    total_c += torch.bincount(labels, minlength=num_classes) # count how many labels there are for each class
                    correct_c += torch.bincount(labels[correct_mask], minlength=num_classes) # count correct predictions for each class
                else:
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        # Compute accuracy
        if self.args.class_accuracy:
            # compute accuracy for each class
            accuracy_c = 100 * correct_c / total_c

            for label in self.train_labels:
                    # label description (for more readable stats)
                    label_desc = f'{label} ({loader.dataset.label_description(label)})'

                    # plot and print new accuracy for this class
                    self.writer.add_scalar(
                        'Accuracy/' + label_desc + ' ' + subset_log,
                        accuracy_c[label],
                        self.absolute_iter(epoch, i)
                    )
                    print(f'{subset_log} accuracy for class {label_desc}: {accuracy_c[label].item():.2f} %')

            # compute global accuracy
            accuracy = 100 * correct_c.sum().item() / total_c.sum().item()
        else:
            # compute only global accuracy
            accuracy = 100 * correct / total

        # Plot new global accuracy point
        self.writer.add_scalar(
            'Global Accuracy/' + subset_log,
            accuracy,
            self.absolute_iter(epoch, i)
        )

        # Best validation accuracy updating
        if subset == 'val':
            if accuracy > self.best_accuracy: # improvement
                # best-so-far accuracy update
                self.best_accuracy = accuracy

                # zero the non-improvements counter
                if self.early_stopping_enable:
                    self.non_imp = 0

                # save validation accuracy for each class
                if self.args.class_accuracy:
                    self.val_accuracy_c_model_save = accuracy_c.to('cpu')

                # after an improvement, let's save the model
                self.save_model(epoch, i)
            elif self.early_stopping_enable:
                self.non_imp += 1

        print(f'Accuracy of the network on the {len(loader.dataset)} {subset_log.lower()} images: {accuracy:.2f} %' + 
              (f' [non-improvements: {self.non_imp}/{self.max_non_imp}]' if self.early_stopping_enable and subset == 'val' else ''))
        
        # Finish Evaluation
        self.net.train()
