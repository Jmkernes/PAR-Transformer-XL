import logging
logging.info('-'*10, ' Importing modules ', '-'*10)
import os
import time
import tensorflow as tf
from data_utils import DataManager

from absl import flags
from absl import app

FLAGS = flags.FLAGS

# flags.DEFINE_boolean('use_defaults', False, 'Whether to use the default data build.')

# flags.DEFINE_boolean('download_data', False,
# 'Set to true to download wikitext2 from the web. Should only be run once.')

flags.DEFINE_boolean('from_directory', False, 'Builds the train, validation, and test sets given a directory.\
Directory must be specified in the text_file flag and contain the files train.txt, valid.txt, and test.txt.')

flags.DEFINE_boolean('train_tokenizer', False,
'Set to true to (re)train SentencePiece Tokenizer. Must also specify vocab_size and model_prefix')

flags.DEFINE_string('text_file', 'data/wikitext-2',
'If from_directory=False, this should be the path of a text file.\
 If from_directory=True, this should point to a directory.')

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

flags.DEFINE_integer('seq_len', 32, 'Must be specified with batch_size.', lower_bound=1)

flags.DEFINE_string('tfrecords_directory', 'data/wikitext2_bsz32_seqlen32_tfrecords',
'The directory to write batched tfrecords to.\
 If from_directory=True, this will be a prefix for train/valid/test directories.\
 If from_directory=False, this will be the full directory name.\
 Suggested naming convention: [prefix]_bsz[batch_size]_seqlen[seq_len]_tfrecords_[train/val/test]')


def build_from_file(configs):
    logging.info(f"\n**** Creating dataset from file {configs['text_file']} ****")
    start = time.time()
    dm = DataManager.initialize_from_text(configs)
    logging.info("\n---- Sample from the dataset  ---")
    logging.info(next(iter(dm.dataset)))
    logging.info(f"\nTotal time to process data: {time.time()-start:.2f}s")
    return dm

def build_from_directory(configs):
    in_dir = configs['text_file']
    out_dir = configs['tfrecords_directory']

    if not os.path.isfile(os.path.join(in_dir, 'train.txt')):
        raise FileExistsError("train.txt file does not exist in specified directory.")
    if not os.path.isfile(os.path.join(in_dir, 'valid.txt')):
        raise FileExistsError("valid.txt file does not exist in specified directory.")
    if not os.path.isfile(os.path.join(in_dir, 'test.txt')):
        raise FileExistsError("test.txt file does not exist in specified directory.")

    configs['text_file'] = os.path.join(in_dir, 'train.txt')
    configs['tfrecords_directory'] = out_dir+'_train'
    train_dm = build_from_file(configs)

    configs['train_tokenizer'] = False
    configs['text_file'] = os.path.join(in_dir, 'valid.txt')
    configs['tfrecords_directory'] = out_dir+'_valid'
    valid_dm = build_from_file(configs)

    configs['text_file'] = os.path.join(in_dir, 'test.txt')
    configs['tfrecords_directory'] = out_dir+'_test'
    test_dm = build_from_file(configs)
    return train_dm, valid_dm, test_dm

default_configs = DataManager.get_default_configs()

def main(argv):
    if not os.path.isfile(FLAGS.text_file):
        raise FileExistsError("Invalid path location for text file.")
    if os.path.isdir(FLAGS.tfrecords_directory):
        logging.warning("The specified tfrecords directory already exists.")
        x = input("To proceed and overwrite, enter [y]. To exit enter any key.")
        if x!='y':
            quit()

    configs = {
        'train_tokenizer': FLAGS.train_tokenizer,
        'text_file': FLAGS.text_file,
        'vocab_size': FLAGS.vocab_size,
        'sp_model_prefix': FLAGS.sp_model_prefix,
        'lowercase': FLAGS.lowercase,
        'shards': FLAGS.shards,
        'batch_size': FLAGS.batch_size,
        'seq_len': FLAGS.seq_len,
        'tfrecords_directory': FLAGS.tfrecords_directory
    }

    logging.info(f"\Current configurations: {configs}")
    if FLAGS.from_directory:
        logging.info(f"\nBuidling from directory: {FLAGS.text_file}")
        build_from_directory(configs)
    else:
        logging.info("\nBuilding from file: {FLAGS.text_file}")
        build_from_file(configs)

if __name__=="__main__":
    app.run(main)





    # Default build. Keep for a shell script later
    # configs = DataManager.get_default_configs()
    # configs['train_tokenizer'] = True
    #
    # logging.info("**** Creating the training dataset ****")
    # start = time.time()
    # dm = DataManager.initialize_from_text(configs)
    # logging.info("---- Sample from the train dataset  ---")
    # logging.info(next(iter(dm.dataset)))
    # logging.info(f"Total time to process training data: {time.time()-start:.2f}s")
    #
    #
    # logging.info("**** Creating the validation dataset ****")
    # configs['train_tokenizer'] = False
    # configs['text_file'] = 'data/wikitext-2/valid.txt'
    # configs['tfrecords_directory'] = 'data/wikitext2_bsz32_seqlen32_tfrecords_valid'
    # start = time.time()
    # dm = DataManager.initialize_from_text(configs)
    # logging.info("---- Sample from the validation dataset  ----")
    # logging.info(next(iter(dm.dataset)))
    # logging.info(f"Total time to process validation data: {time.time()-start:.2f}s")
    #
    # logging.info("**** Creating the test dataset ****")
    # configs['text_file'] = 'data/wikitext-2/test.txt'
    # configs['tfrecords_directory'] = 'data/wikitext2_bsz32_seqlen32_tfrecords_test'
    # start = time.time()
    # dm = DataManager.initialize_from_text(configs)
    # logging.info("---- Sample from the test dataset  ----")
    # logging.info(next(iter(dm.dataset)))
    # logging.info(f"Total time to process test data: {time.time()-start:.2f}s")
