import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def extract_pi_weights(model):
    weights = []
    for layer in model.layers:
        if hasattr(layer, 'pi'):
            weight = layer.pi
            weight = weight/tf.reduce_sum(weight)
            weights.append(weight)
    weights = tf.stack(weights, 0).numpy()
    return weights

def stochastic_block_plot(weights, title='PAR-segmentation'):
    num_blocks = len(weights)
    weights = np.transpose(weights)
    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.matshow(weights, cmap='binary', vmin=0, vmax=1)
    ax.set_yticks(range(3))
    ax.set_yticklabels(['Attention', 'Dense', 'Identity'], rotation=0, size=14)
    ax.set_xticks(range(num_blocks))
    ax.set_xticklabels(range(1, num_blocks+1))
    ax.set_ylabel(f'Block type probability',size=14)
    ax.set_xlabel(f'Stochastic block', size=14)
    plt.title(title)

    # add text boxs for max weight values
    for i,j in zip(np.argmax(weights, 0), np.arange(num_blocks)):
        ax.text(j, i, '{:0.2f}'.format(weights[i,j]), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    return ax, weights

def visualize_pi_weights(model):
    weights = extract_pi_weights(model)
    return stochastic_block_plot(weights)

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
#    end = '\r' if step%100 else '\n'
    print(iter_message, bar, time_message, loss_message, end=end)
