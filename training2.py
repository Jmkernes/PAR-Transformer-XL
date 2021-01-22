print("\n~~~~~~~~ Importing Modules ~~~~~~~~\n")

import os
import time
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
from data_utils import DataManager
from utils import print_bar, visualize_pi_weights
from par_model import PARTransformerXL
from par_model import create_lookahead_mask, positional_encoding

from absl import flags
from absl import app

FLAGS = flags.FLAGS

# Get flag for where to save checkpointed models and tensorboard training data
flags.DEFINE_string('model_name', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
'Model name for saving to checkpoints and log files. Defaults to current time.')

# Get data loading flags
flags.DEFINE_string('train_directory', 'data/wikitext2_bsz32_seqlen32_tfrecords_train',
'Path of training dataset tfrecords directory')

flags.DEFINE_string('valid_directory', 'data/wikitext2_bsz32_seqlen32_tfrecords_valid',
'Path of validation dataset tfrecords directory')

flags.DEFINE_string('test_directory', 'data/wikitext2_bsz32_seqlen32_tfrecords_test',
'Path of testing dataset tfrecords directory')

# Get model parameter flags
flags.DEFINE_integer('d_model', 256, 'Embedding dimension. Used in attention layers.')

flags.DEFINE_integer('num_heads', 8, 'Number of heads to use in MultiHeadAttention.')

flags.DEFINE_integer('d_ffn', 1024, 'Dimension of pointwise feed forward networks.', lower_bound=1)

flags.DEFINE_integer('num_layers', 12, 'Number of stochastic blocks/encoder layers.', lower_bound=0)

flags.DEFINE_integer('mem_len', 32, 'Number of previous values to use as memory.')

flags.DEFINE_float('dropout_rate', 0.1, 'Rate to drop units.')

flags.DEFINE_multi_integer('cutoffs', [], 'Cutoffs to use for adaptive softmax layer. Do NOT\
enter the final cutoff (the vocab size). This will be inferred from your sp_model_file.\
Cutoffs may be entered by repated use of --cutoffs=[NUMBER].')

flags.DEFINE_integer('proj_factor', 4, 'Reduction factor of d_model in adaptive softmax for successive clusters')

flags.DEFINE_multi_integer('proj_dims', [], 'Manually set reduction factors. Must match number of clusters.')

flags.DEFINE_integer('warmup_steps', 4000, 'Number of warmup steps for the learning rate.')

flags.DEFINE_float('tau_start', 2.0, 'Initial value for gumbel softmax temperature tau.')

flags.DEFINE_float('tau_end', 0.2, 'Final value for gumbel softmax temperature tau.')

flags.DEFINE_integer('epochs', 20, 'Number of epochs')

flags.DEFINE_boolean('tau_is_trainable', False, 'Set True to let model learn tau.')

flags.DEFINE_string('opt_name', 'adam', 'Available choices are set by the tf.keras.optimizers.get() call.')

def load_datasets(train, val, test):
    """Load the wikitext2 train, validation and test data"""
    print(f"Loading training data from: {train}")
    config = {'tfrecords_directory': train,'sp_model_prefix': 'wiki2_12k'}
    train_dm = DataManager.initialize_from_tfrecord(config)

    print(f"Loading validation data from: {val}")
    config['tfrecords_directory'] = val
    valid_dm = DataManager.initialize_from_tfrecord(config)

    print(f"Loading testing data from: {test}\n")
    config['tfrecords_directory'] = test
    test_dm = DataManager.initialize_from_tfrecord(config)

    return train_dm, valid_dm, test_dm

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def main(argv):
    # Take care of some flags logic beyond simple constraints.
    if FLAGS.d_model%FLAGS.num_heads:
        raise ValueError('Number of heads must divide d_model')

    train_dm, valid_dm, test_dm = load_datasets(FLAGS.train_directory, FLAGS.valid_directory, FLAGS.test_directory)

    ## Set global constants inferred from the training data.
    BATCH_SIZE = train_dm.batch_size
    SEQ_LEN = train_dm.seq_len
    VOCAB_SIZE = train_dm.tokenizer.vocab_size().numpy()
    DATASET_SIZE = train_dm.ds_size.numpy()
    MAX_POSITION = max(512, FLAGS.mem_len+SEQ_LEN)

    # Take care of additional constraints on inputs that needed the vocab size
    if any([z>=VOCAB_SIZE for z in FLAGS.cutoffs]) or len(set(FLAGS.cutoffs))!=len(FLAGS.cutoffs):
        raise ValueError("Cutoffs must not exceed {VOCAB_SIZE} or contain duplicates.")
    if FLAGS.cutoffs:
        FLAGS.cutoffs.sort()
        FLAGS.cutoffs.append(VOCAB_SIZE)

    ### Define learning rate schedule and simulated annealing schedule for gumbel softmax temperature tau.
    print(f"\nInitializing {FLAGS.opt_name} optimizer with {FLAGS.warmup_steps} warmup steps.")
    learning_rate = CustomSchedule(FLAGS.d_model, FLAGS.warmup_steps)
    optimizer = tf.keras.optimizers.get(FLAGS.opt_name)
    optimizer.learning_rate = learning_rate

    if FLAGS.tau_is_trainable:
        print(f"\nInitializing exponential tau decay: {FLAGS.tau_start}-->{FLAGS.tau_end}.\n")
    tau = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=FLAGS.tau_start,
        decay_steps=DATASET_SIZE*FLAGS.epochs,
        decay_rate=FLAGS.tau_end
    )

    # Setup the model
    tf.keras.backend.clear_session()
    config = {
        'd_model':FLAGS.d_model,
        'num_heads': FLAGS.num_heads,
        'max_position': MAX_POSITION,
        'd_ffn': FLAGS.d_ffn,
        'num_layers': FLAGS.num_layers,
        'mem_len': FLAGS.mem_len,
        'vocab_size': VOCAB_SIZE,
        'dropout_rate': FLAGS.dropout_rate,
        'cutoffs': FLAGS.cutoffs,
        'proj_factor': FLAGS.proj_factor,
        'proj_dims': FLAGS.proj_dims,
    }
    print("\nInitializing model...")
    print("Model parameters:")
    print(config)
    pos_enc = positional_encoding(MAX_POSITION, FLAGS.d_model)
    lookahead_mask = create_lookahead_mask(MAX_POSITION, MAX_POSITION)
    model = PARTransformerXL(**config)

    # Build model by feeding in sample training data
    x_temp, y_temp = next(iter(train_dm.get_inp_tar_pairs()))
    model(x_temp, None, labels=y_temp, training=False)

    # make tau untrainable
    if not FLAGS.tau_is_trainable:
        for layer in model.layers:
            if hasattr(layer, 'tau'):
                layer.tau = tf.cast(tf.constant(1.), tf.float32)

    # print out model summary
    print("\nModel summary:")
    print(model.summary())

    # Define metrics
    train_loss = tf.keras.metrics.Mean()
    valid_loss = tf.keras.metrics.Mean()
    train_perp = tf.keras.metrics.Mean()
    valid_perp = tf.keras.metrics.Mean()

    print("\nDefining training and evaluation steps...")
    # Define the training and evaluation steps via tf functions
    @tf.function
    def train_step(inp, x_mems, labels, tau):
        with tf.GradientTape() as tape:
            loss, mems = model(inp, x_mems=x_mems, labels=labels, training=True, tau=tau)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)
        train_perp(tf.math.exp(loss))
        return mems

    @tf.function
    def evaluation_step(x, x_mems, labels, tau):
        loss, mems = model(x, x_mems=x_mems, labels=labels, tau=tau, training=False)
        valid_loss(loss)
        valid_perp(tf.math.exp(loss))
        return mems

    def evaluation(dataset, tau):
        x_mems = None
        for x, lbl in dataset:
            x_mems = evaluation_step(x, x_mems, lbl, tau)

    # Set up TensorBoard
    print("\nInitializing TensorBoard...")
    train_log_dir = './logs/' + FLAGS.model_name + '/train'
    test_log_dir = './logs/' + FLAGS.model_name + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # # TODO: FIGURE OUT WHAT TO DO HERE, HOW TO LOAD TensorBoard
    # Maybe try os.system()? Or push this into another loading script? idk
    # os.system('tensorboard --logdir ./logs')

    # Configure datasets for training
    print("\nConfiguring datasets for training. Caching, prefetching...")
    glob_step = tf.Variable(0, dtype=tf.int64) # This will break tf.summary if we use int32
    train_ds = train_dm.get_inp_tar_pairs().cache().prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_dm.get_inp_tar_pairs().prefetch(tf.data.AUTOTUNE)
    iterator=iter(train_ds)

    # Set up checkpointing to periodically save the model every epoch
    checkpoint_path = "./checkpoints/train/"+FLAGS.model_name
    print(f"\nInitializing checkpoints. Models will be saved to {checkpoint_path}")
    ckpt = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        glob_step=glob_step,
        iterator=iterator
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        try:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')
        except:
            print("Model may have changed, could not restore checkpoint.")


    # Run the actual training loop!
    print("\n~~~~~~~~~~ Beginning training ~~~~~~~~~~")
    for epoch in range(FLAGS.epochs):

        print('-'*10,f' Epoch {epoch+1} ', '-'*10)
        start = time.time()
        for x in [train_loss, valid_loss, train_perp, valid_perp]:
            x.reset_states()
        mems = None

        for step, (inp, lbl) in enumerate(train_ds):

            mems = train_step(inp, mems, lbl, tau(glob_step))

            diff = (time.time()-start)/(step+1)
            print_bar(step, DATASET_SIZE, diff, train_loss.result().numpy())

            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=glob_step)
                tf.summary.scalar('train_perp', train_perp.result(), step=glob_step)
                tf.summary.scalar('tau', tau(glob_step), step=glob_step)
            glob_step.assign_add(1)

        evaluation(valid_ds, tau(glob_step))
        with test_summary_writer.as_default():
            tf.summary.scalar('valid_loss', valid_loss.result(), step=glob_step)
            tf.summary.scalar('valid_perp', valid_perp.result(), step=glob_step)

        if epoch > 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')


if __name__=="__main__":
    app.run(main)
