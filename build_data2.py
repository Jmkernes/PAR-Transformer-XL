import os
from absl import flags
from absl import app

FLAGS = flags.FLAGS

# flags.DEFINE_boolean('use_defaults', False, 'Whether to use the default data build.')

flags.DEFINE_boolean('download_data', False,
'Set to true to download wikitext2 from the web. Should only be run once.')

flags.DEFINE_boolean('train_tokenizer', False,
'Set to true to (re)train SentencePiece Tokenizer. Must also specify vocab_size and model_prefix')

flags.DEFINE_string('text_file', 'data/wikitext-2/train.txt',
'File path of the text file dataset')

flags.DEFINE_integer('vocab_size', 12000,
'Vocabulary size of tokenizer. Will be ignored if \'train_tokenizer\'=False')

flags.DEFINE_string('sp_model_prefix', 'wiki2_12k',
'SentencePiece builds a *.model and *.vocab file, this is the * prefix.\
 If \'train_tokenizer\'=False, the program will search the current directory\
 for the specified prefix, and return an error if it fails to find it.')

 flags.DEFINE_boolean('lowercase', True, 'Ignored if not training tokenizer')

 flags.DEFINE_integer('shards', 1, 'Number of shards to split tokenized\
  text dataset into when writing to tfrecords.', lower_bound=1)

 flags.DEFINE_integer('batch_size', 32, 'You must specify both batch_size and\
 sequence length (seq_len). The data manager will clean and tokenize the text\
 and batch it to the appropriate sizes.', lower_bound=1)

 flags.DEFINE_integer('seq_len', 32, 'Must be specified with batch_size.',
 lower_bound=1)

 flags.DEFINE_string('tfrecords_directory', 'data/wikitext2_bsz32_seqlen32_tfrecords_train',
 'The directory to write batched tfrecords to. The suggested naming convention is\
  [prefix]_bsz[batch size]_seqlen[sequence length]_tfrecords_[train/val/test]')


def build_dataset(configs):
    Print(f"\n**** Creating dataset from file {configs['text_file']} ****")
    start = time.time()
    dm = DataManager.initialize_from_text(configs)
    print("\n---- Sample from the dataset  ---")
    print(next(iter(dm.dataset)))
    print(f"\nTotal time to process data: {time.time()-start:.2f}s")
    return dm

default_configs = DataManager.get_default_configs()

if __name__=="__main__":

    if not os.path.isfile(FLAGS.text_file):
        raise FileExistsError("Invalid path location for text file.")
    if os.path.isdir(FLAGS.tfrecords_directory):
        x = input("""WARNING: the specified tfrecords directory already exists.
        To exit, enter any key. To proceed and overwrite enter [y].""")
        if x!='y':
            quit()

    print('-'*10, ' Importing modules ', '-'*10)
    import time
    import tensorflow as tf
    from data_utils import DataManager

    configs = {
        'train_tokenizer': FLAGS.train_tokenizer,
        'text_file': FLAGS.text_file,
        'vocab_size': FLAGS.vocab_size,
        'sp_model_prefix': FLAGS.sp_model_prefix,
        'lowercase': FLAGS.lowercase,
        'shards': FLAGS.shards,
        'batch_size': FLAGS.batch_size,
        'seq_len': FLAGS.seq_len,
        'tfrecords_directory': Flags.tfrecords_directory
    }

    build_dataset(configs)





    # Default build. Keep for a shell script later
    # configs = DataManager.get_default_configs()
    # configs['train_tokenizer'] = True
    #
    # print("**** Creating the training dataset ****")
    # start = time.time()
    # dm = DataManager.initialize_from_text(configs)
    # print("---- Sample from the train dataset  ---")
    # print(next(iter(dm.dataset)))
    # print(f"Total time to process training data: {time.time()-start:.2f}s")
    #
    #
    # print("**** Creating the validation dataset ****")
    # configs['train_tokenizer'] = False
    # configs['text_file'] = 'data/wikitext-2/valid.txt'
    # configs['tfrecords_directory'] = 'data/wikitext2_bsz32_seqlen32_tfrecords_valid'
    # start = time.time()
    # dm = DataManager.initialize_from_text(configs)
    # print("---- Sample from the validation dataset  ----")
    # print(next(iter(dm.dataset)))
    # print(f"Total time to process validation data: {time.time()-start:.2f}s")
    #
    # print("**** Creating the test dataset ****")
    # configs['text_file'] = 'data/wikitext-2/test.txt'
    # configs['tfrecords_directory'] = 'data/wikitext2_bsz32_seqlen32_tfrecords_test'
    # start = time.time()
    # dm = DataManager.initialize_from_text(configs)
    # print("---- Sample from the test dataset  ----")
    # print(next(iter(dm.dataset)))
    # print(f"Total time to process test data: {time.time()-start:.2f}s")
