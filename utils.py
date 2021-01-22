import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def visualize_pi_weights(model):
    weights = []
    for layer in model.layers:
        if hasattr(model, 'pi'):
            weight = layer.pi
            weight /= tf.reduce_sum(weight)
            weights.append(layer.pi)
    num_blocks = len(weights)
    weights = tf.stack(weights, 0)

    fig = plt.figure(figsize=(8, 5))
    plt.matshow(weights, cmap='viridis')
    plt.xticks(range(3), ['Attention', 'Dense', 'Identity'], rotation=15, size=14)
    plt.yticks(range(num_blocks), range(1, num_blocks+1))
    plt.xlabel(f'Block type probability', size=16)
    plt.ylabel(f'Stochastic block', size=16)
    plt.show()


def print_bar(step, tot, diff, loss):
    num_eq = int(10*(step+1)/tot)
    num_pd = 10-num_eq
    bar = '['+'='*num_eq+'>'+'.'*num_pd+']'
    time_left = (tot-step)*diff
    m = int(time_left)//60
    s = int(time_left)%60
    iter_message = f"Iteration {step+1:02d}/{tot}:"
    time_message = f"{1/diff:.2f} it/s. Est: {m:02d}m {s:02d}s"
    loss_message = f"Loss: {loss:.3f}"
    end = '\r' if step<tot-1 else '\n'
    print(iter_message, bar, time_message, loss_message, end=end)
