print('\n\n---- Importing modules ---\n')

import time
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
from data_utils import DataManager
from vanilla_transformer import VanillaTransformer
from par_model import create_lookahead_mask, positional_encoding

def printBar(step, tot, diff, loss):
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

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

print('\n\n---- Loading dataset ----')
config = {
  'sp_model_prefix':'wiki2_12k',
  'tfrecords_directory': 'data/wikitext2_bsz32_seqlen32_tfrecords_train'
}
dm = DataManager.initialize_from_tfrecord(config)
train_ds = dm.get_inp_tar_pairs()

learning_rate = CustomSchedule(128, 4000)



tf.keras.backend.clear_session()

config = {
    'd_model':256,
    'num_heads':4,
    'd_ffn':1024,
    'num_layers':12,
    'vocab_size':12000,
    'max_position':512,
    'dropout_rate':0.1,
    'cutoffs':[500, 2500, 12000],
    'proj_factor':4,
    'proj_dims':None,
}

mpos = config['max_position']
pos_enc = positional_encoding(mpos, config['d_model'])
lookahead_mask = create_lookahead_mask(mpos, mpos)
R = pos_enc[:,:32,:]
mask = lookahead_mask[:,:,:32,:32]

print('\n\n---- Configuring model ----')
model = VanillaTransformer(**config)

train_loss = tf.keras.metrics.Mean()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


@tf.function(input_signature=[
    tf.TensorSpec(shape=(32,32), dtype=tf.int32),
     tf.TensorSpec(shape=(32,32), dtype=tf.int32)
])
def train_step(inp, labels):
    with tf.GradientTape() as tape:
        loss = model(inp, position=R, labels=labels, mask=mask)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    try:
      ckpt.restore(ckpt_manager.latest_checkpoint)
      print ('Latest checkpoint restored!!')
    except:
      print("Model may have changed, could not restore checkpoint.")

EPOCHS = 3
tot = 2827
history = {'loss':[]}

if __name__=="__main__":
    for epoch in range(EPOCHS):
        print('-'*10,f' Epoch {epoch+1} ', '-'*10)
        start = time.time()
        train_loss.reset_states()
        for step, (inp, lbl) in enumerate(train_ds):
            train_step(inp, lbl)
            loss = train_loss.result().numpy()
            history['loss'].append(loss)
            diff = (time.time()-start)/(step+1)
            printBar(step, tot, diff, loss)
        plt.plot(history['loss'], 'k-')
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.show()
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
