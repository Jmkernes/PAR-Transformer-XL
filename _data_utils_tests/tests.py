import os
import unittest
from data_utils import *
import tensorflow as tf

class TestDataUtilsFunctions(unittest.TestCase):

    def test_build_and_load_tokenizer(self):
        input_file = 'sample.txt'
        model_prefix = '__temp'
        vocab_size = 250
        lowercase = True
        sp_model_file = build_sentencepiece_model(input_file, model_prefix, vocab_size, lowercase)
        print("\n\n----- Successfully created tokenizer -----\n\n")
        os.remove('__temp.model')
        os.remove('__temp.vocab')
        self.assertTrue(True)

    def test_tokenize_dataset(self):
        text_ds = tf.data.TextLineDataset('sample.txt')
        tokenizer = load_sentencepiece_model('sp250.model')
        tokenized_ds = text_ds.map(tokenizer.tokenize)
        print(next(iter(tokenized_ds)))
        self.assertTrue(True)

    def test_context_batch_dataset(self):
        text_ds = tf.data.TextLineDataset('sample.txt')
        tokenizer = load_sentencepiece_model('sp250.model')
        tokenized_ds = text_ds.map(tokenizer.tokenize)

        batch_size, seq_len = 3, 4
        flat_ds = flatten_dataset_to_tensor(tokenized_ds)
        flat_ds = tf.data.Dataset.from_tensor_slices(flat_ds)
        ds, n_batches, n_tokens = context_batch(flat_ds, batch_size, seq_len)
        self.assertTrue(next(iter(ds)).shape==(batch_size, seq_len))
        print(f"Number of batches: {n_batches}. Number of tokens: {n_tokens}")

        print("\n\n----- Writing to an individual tfrecord -----\n\n")
        name = '__temp.tfrecords'
        write_to_tfrecord(ds, name, n_batches)
        os.remove(name)
        self.assertTrue(True)

    def test_read_individual_tfrecord(self):
        print("\n\n----- Reading from an individual tfrecord -----\n\n")
        read_ds = read_from_tfrecord('one_record.tfrecords')
        self.assertTrue( next(iter(read_ds)).shape == (3, 4) )

    def test_write_to_multiple_tfrecords(self):
        print("\n\n----- Writing to multiple tfrecords -----\n\n")
        text_ds = tf.data.TextLineDataset('sample.txt')
        tokenizer = load_sentencepiece_model('sp250.model')
        tokenized_ds = text_ds.map(tokenizer.tokenize)

        shards = 2
        directory = '__temp_data'
        batch_size, seq_len = 3, 4
        write_to_tfrecord_shards(directory, tokenized_ds, batch_size, seq_len, shards)
        os.remove(os.path.join(directory, 'file1.tfrecords'))
        os.remove(os.path.join(directory, 'file2.tfrecords'))
        os.rmdir(directory)
        self.assertTrue(True)

    def test_reading_from_multiple_tfrecords(self):
        print("\n\n----- Reading from a tfrecord directory -----\n\n")
        read_ds = load_tfrecord_ds_from_files('tfrecords')
        self.assertTrue( next(iter(read_ds)).shape == (3, 4) )

    def test_DataManager_load_from_tfrecords(self):
        configs = {
            'tfrecords_directory':'tfrecords',
             'sp_model_prefix':'sp250'
        }
        dm = DataManager.initialize_from_tfrecord(configs)
        print("\n\n----- Batch initialized from tfrecords -----\n\n")
        print(next(iter(dm.dataset)))
        self.assertTrue(True)

    def test_DataManager_load_from_text(self):
        configs = {
            'train_tokenizer': False,
            'text_file': 'sample.txt',
            'vocab_size': 250,
            'sp_model_prefix': 'sp250',
            'lowercase': True,
            'shards': 1,
            'batch_size': 4,
            'seq_len': 8,
            'tfrecords_directory': '_temp_bsz4_seqlen8_tfrecords',
        }
        try:
            dm = DataManager.initialize_from_text(configs)
            print("\n\n----- Batch initialized from text file -----\n\n")
            print(next(iter(dm.dataset)))
            self.assertTrue(True)
        except:
            os.remove('_temp_bsz4_seqlen8_tfrecords/file1.tfrecords')
            os.rmdir('_temp_bsz4_seqlen8_tfrecords')
            self.assertTrue(False)

if __name__=="__main__":
    unittest.main()
