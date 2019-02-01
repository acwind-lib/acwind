"""
.. module:: nnmodel

This module contains a class wrapper that simplifies the interface for training
a neural network.

.. moduleauthor:: amartinsson, ZofiaTr, cmatthews
"""

from __future__ import absolute_import
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Optimizer
from .helpers import Average

class SupervisedModel():
    """ Class for training of a neural network for classification.

    :param model: torch.nn.Module, acwind.neural_network.FNN or
                  acwind.neural_network.CNN
    :param criterion: torch.nn loss function,
                      for example nn.CrossEntropyLoss()
    :param optimizer: torch.optim optimizer,
                      for example optim.Adam(net.parameters(), lr=0.01)
    :param trainloader: torch.utils.data.DataLoader(), dataset used to train
    :param return_index: True if torch.Dataset returns inputs, label, index,
                         and False if the return is inputs, labelself. The \
                         same as return_index in \
                         acwind.torchhelpers.FeedForwardDataset.

    Example::

        >>> import torch.optim as optim
        >>> import torch.nn as nn
        >>> import acwind.neural_network as acwnn
        >>> net = acwnn.CNN(nFEATURES, nLABELS, L_SIZE=6, C_SIZE=32)
        >>> criterion = nn.CrossEntropyLoss()
        >>> optimizer = optim.Adam(net.parameters(), lr=0.01)
        >>> tr = acwnn.SupervisedModel(net, criterion, optimizer, trainloader, \
        return_index=False)
        >>> tr.fit(1000)
        >>> predictions, Y_true, dataset, proba = tr.predict(trainloader, \
        return_data_set=True, return_probabilities=True)
    """

    def __init__(self, model, criterion, optimizer, trainloader,
                 return_index=True):
        self.net = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.return_index = return_index
        self.fitted_state = None
        self.sampler = None

    def _step(self, element, optimizer):

        # get features and labels
        if self.return_index:
            inputs, label, index = element
        else:
            inputs, label = element
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.net(inputs)

        # get index of the class label
        _, indices = torch.max(label, 1)

        # calculate and update the loss
        loss = self.criterion(outputs, indices)
        loss.backward()
        optimizer.step()

        return loss

    def fit(self, num_epochs, printevery=200):
        """
        Train the neural network.

        :param int num_epochs: number of epochs to run
        :param int printevery: how often to print progress
        """
        ####------------------ Train on Dataset ------------------####
        for epoch in range(num_epochs):

            running_loss = 0.0

            for i, element in enumerate(self.trainloader, 0):

                # optimizer step
                loss = self._step(element, self.optimizer)

                # print statistics
                running_loss += loss.item()
                # print every _printevery_ mini-batches
                if i % printevery == (printevery-1):
                    print('[%2d, %4d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / printevery))
                    running_loss = 0.0

        self.fitted_state = self.net.state_dict()
        print('Finished Training')

    def predict(self, testloader_1,
                return_data_set=False, local_series=False,
                return_index=None, return_probabilities=False,
                print_accuracy=True):
        """
        Prediction for the trained network.

        :param testloader_1: torch.utils.data.DataLoader(),
                             dataset used to predict
        :param return_data_set: return dataset
        :param return_index: return indices of the dataset. Requires the
                             dataloader to return inputs, label, index
                             in __getitem__(self, index)
        :param return_probabilities: return list of the probabilities for the
                                     predicted class for the dataset
        :return: list of predictions and true labels, optionally dataset and
                 indices
        """

        dataset = []
        true = []
        predictions = []
        probabilities = []

        if return_index is None:
            return_index = self.return_index

        if return_index:
            indices = []

        with torch.no_grad():
            for i, element in enumerate(testloader_1, 0):

                # get the data
                if return_index:
                    inputs, label, index = element
                else:
                    inputs, label = element

                if local_series:
                    x = inputs.data.numpy()[:, :, 1, 1]
                else:
                    x = inputs.data.numpy()
                #lb = label.data.numpy()

                dataset += [copy.deepcopy(x)]

                # get the output
                outputs = self.net(inputs)

                # predicted class
                _, predicted = torch.max(outputs.data, 1)
                _, truth = torch.max(label, 1)

                true = np.append(true, truth, axis=0)
                predictions = np.append(
                    predictions, predicted.data.numpy(), axis=0)

                if return_index:
                    indices = np.append(indices, index.data.numpy(), axis=0)

                if return_probabilities:
                    probabilities += [torch.softmax(outputs, 1).data.numpy()]

        if print_accuracy:
            print('Accuracy of the network on the test set: %d %%' %
                  (100 * sum(true == predictions)/len(predictions)))

        return_list = [predictions, true]

        if return_data_set:
            return_list.append(np.vstack(dataset))

        if return_index:
            return_list.append(np.hstack(indices))

        if return_probabilities:
            return_list.append(np.vstack(probabilities))

        return return_list

    def save_network(self, folder_name):
        """
        Save trained network.

        :param folder_name str: to be saved as folder_name,
                                recommended as 'folder_name.pt'
        """
        torch.save(self.net.state_dict(), folder_name)

    def load_network(self, folder_name=None, state_dict=None):
        """
        Load saved trained network.

        :param folder_name str: folder name in format '.pt'
        :param state_dict dict: previously used state dictionary
        """

        if state_dict is not None:
            self.net.load_state_dict(state_dict)
        else:
            self.net.load_state_dict(torch.load(folder_name))
