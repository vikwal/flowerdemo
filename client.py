import argparse
import os
import json
import numpy as np
import pandas as pd

import flwr as fl
import tensorflow as tf
from tensorflow import keras


# Hyperparameters
server_address = '192.168.240.4:8080'
verbose = 0

# Arguments
parser = argparse.ArgumentParser(description='flowerdemo client skript')
parser.add_argument('client_nr', type=int, help='Client number for partitioning train and test data', choices=(range(0,4)))
args = parser.parse_args()

# laod data
client_nr = args.client_nr
print('Client nr:', client_nr)

df = pd.read_csv('mnist_for_client_'+str(client_nr)+'.csv')

train_set = df[:12500]
test_set = df[12500:]

x_train, y_train = train_set.drop('y', axis=1).values.reshape(-1,28,28), train_set.y
x_test, y_test = test_set.drop('y', axis=1).values.reshape(-1,28,28), test_set.y

# Define Flower client
class Client(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        model_config = json.loads(config['model']) # use json.loads() to convert it back to dict type
        self.model = keras.Model().from_config(model_config)
        self.model.compile(optimizer=config['optimizer'], loss=config['loss'], metrics=config['metrics'])

        self.model.set_weights(parameters)

        batch_size: int = config['batch_size']
        epochs: int = config['local_epochs']

        self.model.set_weights(parameters)
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return self.model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

#log output
fl.common.logger.configure(identifier='flowerdemo client nr:'+str(client_nr), filename='client_'+str(client_nr)+'.txt')

# Start Flower client
fl.client.start_numpy_client(server_address=server_address, client=Client(None))
