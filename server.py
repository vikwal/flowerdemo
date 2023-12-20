import json
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays
import tensorflow as tf
from tensorflow import keras

# Hyperparameters
# fl-specific
server_address = '0.0.0.0:8080'
num_rounds = 3
min_fit_clients = 2
min_available_clients = 2
# model-specific
optimizer = 'adam'
loss = 'sparse_categorical_crossentropy'
metrics = 'accuracy'
batch_size = 32
local_epochs = 2

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggr_weights, aggr_metrics = super().aggregate_fit(rnd, results, failures)
        self.aggr_weights = parameters_to_ndarrays(aggr_weights)
        return aggr_weights, aggr_metrics

def fit_config(rnd: int):
    config = {
        "batch_size": batch_size,
        "local_epochs": local_epochs,
        "model": model_config,
        "optimizer": optimizer, 
        "loss": loss, 
        "metrics": metrics
    }
    return config

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m['accuracy'] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {'accuracy': sum(accuracies) / sum(examples)}

# Model
input = keras.layers.Input(shape=(28,28))
x = keras.layers.Flatten(input_shape=(28,28))(input)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dense(256, activation='relu')(x)
output = keras.layers.Dense(10, activation='softmax')(x)

model_arch = keras.Model(input, output)
model = model_arch 

model.compile(optimizer, loss, metrics=[metrics])

model_config = json.dumps(model.get_config())

# Create strategy
strategy = SaveModelStrategy(
    min_fit_clients=min_fit_clients,
    min_available_clients=min_available_clients,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average,
    initial_parameters=ndarrays_to_parameters(model.get_weights()),
)

#log output
fl.common.logger.configure(identifier='flowerdemo server', filename='server.txt')

# Start Flower server
fl.server.start_server(
    server_address=server_address,
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy
)
