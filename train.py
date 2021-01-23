import logging
logging.info("\n\n~~~~~~~~ Importing Modules ~~~~~~~~\n")

import os
import json
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

### Checkpointing and TensorBoarding parameter flag(s)
flags.DEFINE_string('model_name',
    default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    help='Model name for saving to checkpoints and log files. \
         Defaults to current time.')

### Data path locaation flags
flags.DEFINE_string('train_directory',
    default='data/wikitext2_bsz32_seqlen32_tfrecords_train',
    help='Path of training dataset tfrecords directory')
flags.DEFINE_string('valid_directory',
    default='data/wikitext2_bsz32_seqlen32_tfrecords_valid',
    help='Path of validation dataset tfrecords directory')
flags.DEFINE_string('test_directory',
    default='data/wikitext2_bsz32_seqlen32_tfrecords_test',
    help='Path of testing dataset tfrecords directory')

### Get model parameter flags
flags.DEFINE_integer('d_model', default=256,
    help='Embedding dimension. Used in attention layers.')
flags.DEFINE_integer('num_heads', default=8,
    help='Number of heads to use in MultiHeadAttention.')
flags.DEFINE_integer('d_ffn', default=1024,
    help='Dimension of pointwise feed forward networks.',
    lower_bound=1)
flags.DEFINE_integer('num_layers', default=12,
    help='Number of stochastic blocks/encoder layers.',
    lower_bound=0)
flags.DEFINE_integer('mem_len', default=32,
    help='Number of previous values to use as memory.')
flags.DEFINE_float('dropout_rate', default=0.1,
    help='Rate to drop units.')
flags.DEFINE_multi_integer('cutoffs', default=[],
    help='Cutoffs to use for adaptive softmax layer. Do NOT\
         enter the final cutoff (the vocab size). This will \
         be inferred from your sp_model_file. Cutoffs may be \
         entered by repated use of --cutoffs=[NUMBER].')
flags.DEFINE_integer('proj_factor', default=4,
    help='Reduction factor of d_model in adaptive softmax for successive clusters')
flags.DEFINE_boolean('straight_through', default=False,
    help='Set True to enable straight_through gradient in RelaxedOneHot layer.')
flags.DEFINE_multi_integer('proj_dims', default=[],
    help='Manually set reduction factors. Must match number of clusters.')

### Learning parameters
flags.DEFINE_float('max_lr', default=1e-4,
    help='Maximum learning rate after warmup. Used in CosineSchedule.')
flags.DEFINE_integer('warmup_steps', default=4000,
    help='Number of warmup steps for the learning rate.')
flags.DEFINE_float('tau_start', default=2.0,
    help='Initial value for gumbel softmax temperature tau.')
flags.DEFINE_float('tau_end', default=0.2,
    help='Final value for gumbel softmax temperature tau.')
flags.DEFINE_integer('epochs', default=20,
    help='Number of epochs')
flags.DEFINE_boolean('tau_is_trainable', default=False,
    help='Set True to let model learn tau.')
flags.DEFINE_string('opt_name', default='adam',
    help='Available choices are set by the tf.keras.optimizers.get() call.')


### Data loading functions
def load_datasets(train, val, test):
    """Load the wikitext2 train, validation and test data"""
    logging.info(f"\nLoading training data from: {train}")
    config = {'tfrecords_directory': train,'sp_model_prefix': 'wiki2_12k'}
    train_dm = DataManager.initialize_from_tfrecord(config)

    logging.info(f"\nLoading validation data from: {val}")
    config['tfrecords_directory'] = val
    valid_dm = DataManager.initialize_from_tfrecord(config)

    logging.info(f"\nLoading testing data from: {test}\n")
    config['tfrecords_directory'] = test
    test_dm = DataManager.initialize_from_tfrecord(config)

    return train_dm, valid_dm, test_dm


### Custom learning rate schedulers
class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "warmup_steps": self.warmup_steps}
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class CosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, decay_steps, warmup_steps=4000, **kwargs):
        super().__init__(**kwargs)
        self.max_lr = max_lr
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.pi = 3.1415927
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "warmup_steps": self.warmup_steps}
    def __call__(self, step):
        linear = self.max_lr*(step/self.warmup_steps)
        angle = self.pi*tf.math.maximum(step-self.warmup_steps, 0)/self.decay_steps
        cosine = 0.5*self.max_lr*(1+tf.math.cos(angle))
        return tf.math.minimum(linear, cosine)

def main(argv):
    # Take care of some flags logic beyond simple constraints.
    if FLAGS.d_model%FLAGS.num_heads:
        raise ValueError('Number of heads must divide d_model')

    train_dm, valid_dm, test_dm = load_datasets(FLAGS.train_directory, FLAGS.valid_directory, FLAGS.test_directory)

    ## Set global constants inferred from the training data.
    BATCH_SIZE = int(train_dm.batch_size)
    SEQ_LEN = int(train_dm.seq_len)
    VOCAB_SIZE = int(train_dm.tokenizer.vocab_size())
    DATASET_SIZE = int(train_dm.ds_size.numpy())
    MAX_POSITION = int(max(512, FLAGS.mem_len+SEQ_LEN))

    # Take care of additional constraints on inputs that needed the vocab size
    if any([z>=VOCAB_SIZE for z in FLAGS.cutoffs]) or len(set(FLAGS.cutoffs))!=len(FLAGS.cutoffs):
        raise ValueError(f"Cutoffs must not exceed {VOCAB_SIZE} or contain duplicates.")
    if FLAGS.cutoffs:
        FLAGS.cutoffs.sort() # this is redundant, the layer sorts anyway. but to be safe...
        FLAGS.cutoffs.append(VOCAB_SIZE)

    ### Define learning rate schedule and simulated annealing schedule for gumbel softmax temperature tau.
    logging.info(f"\n\nInitializing {FLAGS.opt_name} optimizer with {FLAGS.warmup_steps} warmup steps.")
    decay_steps = DATASET_SIZE*FLAGS.epochs-FLAGS.warmup_steps
    learning_rate = CosineSchedule(FLAGS.max_lr, FLAGS.warmup_steps, decay_steps) # Max learning rate here
    optimizer = tf.keras.optimizers.get(FLAGS.opt_name)
    optimizer.learning_rate = learning_rate
    optimizer.clipvalue = 0.1

    if FLAGS.tau_is_trainable:
        logging.info(f"\n\nInitializing exponential tau decay: {FLAGS.tau_start}-->{FLAGS.tau_end}.\n")
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
        'straight_through': FLAGS.straight_through
    }
    logging.info("\n\nInitializing model...")
    logging.info("Model parameters:")
    logging.info(config)
    # the below are needed for vanilla_transformer, but not PARtransformer
    # pos_enc = positional_encoding(MAX_POSITION, FLAGS.d_model)
    # lookahead_mask = create_lookahead_mask(MAX_POSITION, MAX_POSITION)
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
    logging.info("\nModel summary:")
    logging.info(model.summary())

    # Define metrics
    train_loss = tf.keras.metrics.Mean()
    valid_loss = tf.keras.metrics.Mean()
    train_perp = tf.keras.metrics.Mean()
    valid_perp = tf.keras.metrics.Mean()

    logging.info("\n\nDefining training and evaluation steps...")
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
    logging.info("\n\nInitializing TensorBoard...")
    train_log_dir = './logs/' + FLAGS.model_name + '/train'
    test_log_dir = './logs/' + FLAGS.model_name + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # # TODO: FIGURE OUT WHAT TO DO HERE, HOW TO LOAD TensorBoard during a script
    # If in colab write:
    # %load_ext tensorboard
    # %tensorboard --logdir logs/

    # Configure datasets for training
    logging.info("\n\nConfiguring datasets for training. Caching, prefetching...")
    glob_step = tf.Variable(0, dtype=tf.int64) # This will break tf.summary if we use int32
    train_ds = train_dm.get_inp_tar_pairs().cache().prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_dm.get_inp_tar_pairs().prefetch(tf.data.AUTOTUNE)
    iterator=iter(train_ds)

    # Set up checkpointing to periodically save the model every epoch
    checkpoint_path = "./checkpoints/train/"+FLAGS.model_name
    logging.info(f"\n\nInitializing checkpoints. Models will be saved to {checkpoint_path}")
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
            logging.info('Latest checkpoint restored!!')
        except:
            logging.warning("Model may have changed, could not restore checkpoint.")
    with open(checkpoint_path+'/config', 'w') as file:
        file.write(json.dumps(config))
    logging.info(f"Writing model configuration to {checkpoint_path+'/config'}")
    logging.info("""To recover this model from checkpoint, initialize a model using the config file.
    Then, create a tf.train.Checkpoint pointing at this directory, set model=model, run the restore,
    method of the checkpoint, then finally the uninitialized model should acquire the weights.""")

    ckpt_save_path = ckpt_manager.save()
    logging.info(f'Checkpointing model initialization at {ckpt_save_path}')


    # Run the actual training loop!
    absolute_start = time.time()
    logging.info("\n\n~~~~~~~~~~ Beginning training ~~~~~~~~~~")
    for epoch in range(FLAGS.epochs):

        logging.info('-'*10+f' Epoch {epoch+1}/{FLAGS.epochs} '+'-'*10)
        start = time.time()
        for x in [train_loss, valid_loss, train_perp, valid_perp]:
            x.reset_states()
        mems = None

        for step, (inp, lbl) in enumerate(train_ds):

            mems = train_step(inp, mems, lbl, tau(glob_step))

            diff = (time.time()-start)/(step+1)
            print_bar(step, DATASET_SIZE, diff, train_loss.result().numpy())

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=glob_step)
                tf.summary.scalar('perplexity', train_perp.result(), step=glob_step)
                tf.summary.scalar('tau', tau(glob_step), step=glob_step)
                tf.summary.scalar('lr', learning_rate(tf.cast(glob_step, tf.float32)),
                              step=glob_step)
            glob_step.assign_add(1)

            if (step+1)%1000==0:
                logging.info("Global step:", glob_step.numpy())
                visualize_pi_weights(model)
                plt.show()

        evaluation(valid_ds, tau(glob_step))
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', valid_loss.result(), step=glob_step)
            tf.summary.scalar('perplexity', valid_perp.result(), step=glob_step)

        ckpt_save_path = ckpt_manager.save()
        logging.info(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    tot_time = time.time()-absolute_start
    minutes = tot_time//60
    seconds = tot_time%60
    logging.info('*'*100)
    logging.info("\n\nTRAINING COMPLETE.\n\n")
    logging.info('*'*100)
    logging.info(f"\n\nTotal time: {minutes:.02d}min. {seconds:.02d}sec.\n\n")
    try:
        os.mkdir('saved_models')
    except:
        pass
    logging.info(f"Saving final model to {'saved_models/'+FLAGS.model_name}")
    model.save('saved_models/'+FLAGS.model_name)

if __name__=="__main__":
    app.run(main)
