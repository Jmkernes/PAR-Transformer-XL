print('-'*10, ' Importing modules', '-'*10)

from data_utils import DataManager
import tensorflow as tf
import time

if __name__=="__main__":
    configs = DataManager.get_default_configs()
    configs['train_tokenizer'] = True

    print("**** Creating the training dataset ****")
    start = time.time()
    dm = DataManager.initialize_from_text(configs)
    print("---- Sample from the train dataset  ---")
    print(next(iter(dm.dataset)))
    print(f"Total time to process training data: {time.time()-start:.2f}s")


    print("**** Creating the validation dataset ****")
    configs['train_tokenizer'] = False
    configs['text_file'] = 'data/wikitext-2/valid.txt'
    configs['tfrecords_directory'] = 'data/wikitext2_bsz32_seqlen32_tfrecords_valid'
    start = time.time()
    dm = DataManager.initialize_from_text(configs)
    print("---- Sample from the validation dataset  ----")
    print(next(iter(dm.dataset)))
    print(f"Total time to process validation data: {time.time()-start:.2f}s")
    
    print("**** Creating the test dataset ****")
    configs['text_file'] = 'data/wikitext-2/test.txt'
    configs['tfrecords_directory'] = 'data/wikitext2_bsz32_seqlen32_tfrecords_test'
    start = time.time()
    dm = DataManager.initialize_from_text(configs)
    print("---- Sample from the test dataset  ----")
    print(next(iter(dm.dataset)))
    print(f"Total time to process test data: {time.time()-start:.2f}s")
