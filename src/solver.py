
import torch
import torch.optim as optim
import torch.nn as nn
import os
from typing import Literal

from model import Net

class Solver(object):
    '''
    Solver for training and validation stages
    '''
    def __init__(self, train_loader, val_loader, device, writer, args):
        self.args = args
        self.model_name = 'model_{}.pth'.format(self.args.model_name)
        self.n_classes = train_loader.dataset.n_classes()
        self.range_labels = train_loader.dataset.range_labels()

        # Define the model
        self.net = Net(self.args, self.n_classes).to(device)

        # Load a pretrained model
        if self.args.resume_train:
            self.load_model()
        
        # Define Loss function
        self.criterion = nn.CrossEntropyLoss() if self.n_classes > 1 else nn.BCEWithLogitsLoss()
        
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

        # Tensorboard writer
        self.writer = writer

    def absolute_iter(self, epoch, i):
        '''
        Returns the absolute iteration from received values
        '''
        return epoch * len(self.train_loader) + i

    def save_model(self, epoch, i):
        '''
        Saves the model
        '''
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        os.makedirs(os.path.dirname(check_path), exist_ok=True) # create dir if it doesn't exist
        torch.save(self.net.state_dict(), check_path)
        self.iter_model_save = self.absolute_iter(epoch, i)
        print('Model saved!')

    def load_model(self):
        '''
        Loads the model
        '''
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        self.net.load_state_dict(torch.load(check_path))
        print('Model loaded!')

    def early_stopping(self) -> bool:
        '''
        Returns True if the early stopping condition is satisfied, False otherwise
        '''
        return (self.non_imp > self.max_non_imp) if self.early_stopping_enable else False
    
    def save_class_accuracy_histogram(self):
        '''
        Saves on tensorboard the class accuracy histogram for the last saved mode
        '''
        for label in self.range_labels:
            self.writer.add_histogram(
                f'Model class accuracy (absolute iter {self.iter_model_save})',
                self.val_accuracy_c_model_save[label].item(),
                label
            )

    def store_info_settings(self):
        '''
        Stores on tensorboard information and settings about model and training
        '''
        table = (
            '| Setting | Value |\n'
            '|---------|-------|\n'
            f'| **Run name** | {self.args.run_name} |\n'
            f'| **Model name** | {self.args.model_name} |\n'
            f'| **Model** | ResNet-{self.args.depth} |\n'
            f'| **Pretrained** | {'🟢 yes' if self.args.pretrained else '🔴 no'} |\n'
            f'| **Freezed modules** | {', '.join(self.args.freeze) if self.args.freeze else '-'} |\n'
            f'| **Optimizer** | {self.args.opt} |\n'
            f'| **Epochs** | {self.args.epochs} |\n'
            f'| **Batch Size** | {self.args.batch_size} |\n'
            f'| **Learning Rate** | {self.args.lr} |\n'
            f'| **Norm layers** | {'🟢 yes' if self.args.use_norm else '🔴 no'} |\n'
            f'| **Early stopping** | {f'🟢 yes (after {self.args.early_stopping} non-improvements on validation)' if self.early_stopping_enable else '🔴 no'} |\n'
            f'| **Classes** |' +
                f'{', '.join(f'{self.train_loader.dataset.label_description(label)} ({label})' for label in self.range_labels) if self.train_loader.dataset.classes_subset_enabled() else 'all'} |'
        )
        self.writer.add_text('Model Info/Settings', table)

    def store_saved_model_performance(self):
        '''
        Stores saved model performance on tensorboard
        '''
        table = (
            '| Metric | Value (%) |\n'
            '|--------|-----------|\n'
            f'| **Global accuracy** | {self.best_accuracy:.2f} |\n'
        )
        if self.args.class_accuracy and self.n_classes > 1:
            for label in self.range_labels:
                label_desc = self.train_loader.dataset.label_description(label)
                acc_value = self.val_accuracy_c_model_save[label].item()
                table += f'| Accuracy for class **{label} ({label_desc})** | {acc_value:.2f} |\n'
        self.writer.add_text('Model Info/Validation performance', table)
    
    def train(self):
        '''
        Trains the model on the training set
        '''
        # Store general info
        self.store_info_settings()

        # TRAINING
        self.net.train()
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                #print(inputs)
                #print(labels)
                
                # Put data on correct device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = self.net(inputs)
                if self.n_classes == 1:
                    outputs = outputs.view(-1) # flatten (from size(1, n) to size(n))
                #print(outputs)
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
        if self.args.class_accuracy and self.n_classes > 1:
            self.save_class_accuracy_histogram()
        
        # Save performance of the saved model
        self.store_saved_model_performance()

        self.writer.flush()
        self.writer.close()

        print('Finished Training' + (' (early stop)' if epoch < self.epochs - 1 else ''))  
    
    def evaluation(self, epoch, i, subset: Literal['train', 'val']='val'):
        '''
        Evaluates the model on the specified data subset
        '''
        # Put net into evaluation mode
        self.net.eval()

        # Log using much
        subset_log = ('Validation' if subset == 'val' else 'Training')

        # Select correct loader
        loader = self.val_loader if subset == 'val' else self.train_loader

        # Global accuracy and, eventually, the accuracy for each class
        if self.args.class_accuracy and self.n_classes > 1:
            correct_c = torch.zeros(self.n_classes).to(self.device)
            total_c = torch.zeros(self.n_classes).to(self.device)
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
                if self.n_classes == 1:
                    outputs = outputs.view(-1) # flatten (from size(1, n) to size(n))

                # prediction
                if self.n_classes > 1:
                    _, predicted = torch.max(outputs.data, 1) # the class with the highest energy is what we choose as prediction
                else:
                    predicted = (outputs.sigmoid() > 0.5).int() # if sigmoid(output) > 0.5 we predict positive

                # correct predictions counting
                if self.args.class_accuracy and self.n_classes > 1:
                    correct_mask = predicted == labels
                    total_c += torch.bincount(labels, minlength=self.n_classes) # count how many labels there are for each class
                    correct_c += torch.bincount(labels[correct_mask], minlength=self.n_classes) # count correct predictions for each class
                else:
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        # Compute accuracy
        if self.args.class_accuracy and self.n_classes > 1:
            # compute accuracy for each class
            accuracy_c = 100 * correct_c / total_c

            for label in self.range_labels:
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
                if self.args.class_accuracy and self.n_classes > 1:
                    self.val_accuracy_c_model_save = accuracy_c.to('cpu')

                # after an improvement, let's save the model
                self.save_model(epoch, i)
            elif self.early_stopping_enable:
                self.non_imp += 1

        print(f'Accuracy of the network on the {len(loader.dataset)} {subset_log.lower()} images: {accuracy:.2f} %' + 
              (f' [non-improvements: {self.non_imp}/{self.max_non_imp}]' if self.early_stopping_enable and subset == 'val' else ''))
        
        # Finish Evaluation
        self.net.train()
