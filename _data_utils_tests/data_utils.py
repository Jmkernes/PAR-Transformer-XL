import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

import pdb

# Train a sentencepiece model
def build_sentencepiece_model(input_file, model_prefix, vocab_size, lowercase):
    import sentencepiece as spm

    if model_prefix is None:
        model_prefix = f"wiki2_{vocab_size}"
    normalization_rule = 'nfkc_cf' if lowercase else 'nfkc'

    print("Training SentencePiece on file: {input_file}")
    start = time.time()
    spm.SentencePieceTrainer.Train(
        f'--input={input_file} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--user_defined_symbols=@-@ '
        f'--normalization_rule_name={normalization_rule} '
        f'--pad_id=0 --unk_id=3 --bos_id=1 --eos_id=2 '
        f'--pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[BOS] --eos_piece=[EOS] '
    )
    print(f"""Finished training. Time: {time.time()-start:.2f}s.
     Model protobuf saved to: {model_prefix}.model.
     Vocab size: {vocab_size}.""")
    return f"{model_prefix}.model"


def load_sentencepiece_model(model_proto):
    proto = tf.io.gfile.GFile(model_proto, 'rb').read()
    return tf_text.SentencepieceTokenizer(model=proto)

def find_dataset_size(dataset):
    return tf.cast(dataset.reduce(0, lambda x, _: x+1), tf.int64)

def get_dataset_type(dataset):
    try:
        return next(iter(dataset)).dtype
    except:
        raise ValueError("Dataset is empty.")

def flatten_dataset_to_tensor(dataset):
    dtype = get_dataset_type(dataset)
    res = tf.constant([], dtype=dtype)
    for x in dataset:
        res = tf.concat([res, tf.reshape(x, -1)], 0)
    return res

def context_batch(dataset, batch_size, seq_len, dataset_size=None):
    """ creates batched dataset of temporally contiguous sequences.
    Returns:
        context batched dataset
        number of batches
        number of tokens"""
    if dataset_size is None:
        dataset_size = find_dataset_size(dataset)
    assert dataset_size >= batch_size, "Dataset is smaller than batch size"
    stride = dataset_size//(batch_size*seq_len)
    dataset = dataset.window(seq_len, seq_len).flat_map(lambda w: w.batch(seq_len))
    dataset = dataset.window(batch_size, 1, stride)
    dataset = dataset.flat_map(lambda w: w.batch(batch_size, drop_remainder=True))
    return dataset, stride, dataset_size # stride is num_batches, dataset_size is num_tokens

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# NOTE: I kept tot_tokens and total_lines descriptors here if needed
def serialize_token_batch(batch):
    serial = tf.io.serialize_tensor(batch)
    feature = {'batch': _bytes_feature(serial)}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def deserialize_token_batch(example_proto, out_type=None):
    """ If using sentencepiece, convert to int32, if using TextVectorizationLayer use int64 """
    out_type = tf.int32 if out_type is None else out_type
    feature_description = {'batch': tf.io.FixedLenFeature([], tf.string, default_value=b'')}
    proto = tf.io.parse_single_example(example_proto, feature_description)
    batch = proto['batch']
    batch = tf.io.parse_tensor(batch, out_type=out_type) # change to 64 for vectorizeLayer
    return batch

def write_to_tfrecord(dataset, filename, dataset_size=None):
    if dataset_size is None:
        dataset_size = find_dataset_size(dataset)
    with tf.io.TFRecordWriter(filename) as writer:
        start = time.time()
        for i, x in enumerate(dataset):
            proto = serialize_token_batch(x)
            writer.write(proto)
            end = '\n' if i == dataset_size-1 else '\r'
            print(f"{100*(i+1)/dataset_size:.2f}% complete", end=end)
        print(f"Writing time: {time.time()-start:.2f}s")

def write_to_tfrecord_shards(directory, dataset, batch_size, seq_len, shards):
    """ WARNING: sharding too many times with a large batch_size and seq_len can lead to
    a large amount of unused data. For N shards, B batch size and T seq_len, we loss <N
    lines from sharding, and within each shard we discard the final batch < B*T. Thus,
    the max amount of tokens discarded is bounded above by < (B*T*N + N*avg_line_len).
    ONLY shard if your dataset is very large and/or cannot fit into memory."""

    print(f"Creating tfrecords directory: {directory}.")
    os.mkdir(directory)
    ds_size = find_dataset_size(dataset)
    shard_size = ds_size//shards
    for i in range(shards):
        filename = os.path.join(directory, f'file{i+1}.tfrecords')
        ds = dataset.skip(i*shard_size).take(shard_size)
        print(f"Loading shard {i+1}...")
        flat_ds = flatten_dataset_to_tensor(ds)
        flat_ds = tf.data.Dataset.from_tensor_slices(flat_ds)
        shard_ds, num_batches, _ = context_batch(flat_ds, batch_size, seq_len)
        print(f"--- Writing shard {i+1} to {filename} ---")
        write_to_tfrecord(shard_ds, filename, num_batches)

def read_from_tfrecord(filename, out_type=None):
    out_type = tf.int32 if out_type is None else out_type
    ds = tf.data.TFRecordDataset(filename)
    ds = ds.map(lambda x: deserialize_token_batch(x, out_type=out_type))
    return ds

def load_tfrecord_ds_from_files(path, out_type=None):
    out_type = tf.int32 if out_type is None else out_type
    if os.path.isdir(path):
        filenames = os.listdir(path)
        filenames = [os.path.join(path, name) for name in filenames]
        file_ds = tf.data.Dataset.from_tensor_slices(filenames)
        ds = file_ds.interleave(lambda x: tf.data.TFRecordDataset(x),
                                num_parallel_calls=tf.data.AUTOTUNE)
    elif os.path.isfile(path):
        ds = tf.data.TFRecordDataset(path)
    else:
        raise ValueError("Invalid path.")
    ds = ds.map(lambda x: deserialize_token_batch(x, out_type=out_type))
    return ds


class DataManager:
    def __init__(self, dataset, tokenizer):
        # self.__dict__.update({k: v for k, v in locals().items() if k != 'self'}) # nice one-liner that auto assigns everything in constructor to an attribute
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.ds_size = find_dataset_size(dataset)
        z = next(iter(dataset))
        self.batch_size = z.shape[0]
        self.seq_len = z.shape[1]-1

    @staticmethod
    def get_default_configs():
        configs = {
            'text_file': 'data/wikitext-2/train.txt',
            'vocab_size': 15000,
            'sp_model_prefix': 'wiki2_'+str(15)+'k',
            'lowercase': True,
            'shards': 1,
            'batch_size': 32,
            'seq_len': 32,
            'tfrecords_directory': 'wikitext2_bsz32_seqlen32_tfrecords'
        }

    @classmethod
    def initialize_from_text(cls, configs):
        # train the sentence piece model
        text_file, vocab_size = configs['text_file'], configs['vocab_size']
        sp_model_prefix, lowercase = configs['sp_model_prefix'], configs['lowercase']
        if configs['train_tokenizer']:
            if not os.path.isfile(input_file):
                raise FileExistsError("Could not locate file.")
            sp_model_file = build_sentencepiece_model(text_file, sp_model_prefix, vocab_size, lowercase)
        else:
            sp_model_file = configs['sp_model_prefix']+'.model'

        # load the text dataset and tokenizer
        text_ds = tf.data.TextLineDataset(text_file)
        tokenizer = load_sentencepiece_model(sp_model_file)
        tokenized_ds = text_ds.map(tokenizer.tokenize)

        # Write the shards to tfrecords files
        # Plus one because we will drop one token for the training task!
        ds_size = find_dataset_size(tokenized_ds)
        batch_size, seq_len, shards = configs['batch_size'], configs['seq_len']+1, configs['shards']
        directory = configs['tfrecords_directory']
        write_to_tfrecord_shards(directory, tokenized_ds, batch_size, seq_len, shards)

        tfrecord_configs = {'tfrecords_directory': directory,
        'sp_model_prefix': sp_model_prefix}
        return cls.initialize_from_tfrecord(tfrecord_configs)

    @classmethod
    def initialize_from_tfrecord(cls, configs):
        sp_model_file = configs['sp_model_prefix']+'.model'
        print(f"Loading tokenizer from {sp_model_file}...")
        tokenizer = load_sentencepiece_model(sp_model_file)
        print("Loading tfrecords from directory")
        dataset = load_tfrecord_ds_from_files(configs['tfrecords_directory'])
        new_configs = {'dataset': dataset, 'tokenizer': tokenizer}
        return cls(**new_configs)

    def get_inp_tar_pairs(self):
        return self.dataset.map(lambda x: (x[:, :-1], x[:,1:]))
