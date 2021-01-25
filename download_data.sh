#!/bin/sh

DATA_ROOT=./data
TEXT_DIR=${DATA_ROOT}/wikitext-2
TFRECORD_DIR=${DATA_ROOT}/wiki2_bsz32_seqlen32_tfrecords
FROM_DIRECTORY=True
VOCAB_SIZE=12000
SP_MODEL_PREFIX=wiki2_12k
LOWERCASE=True
SHARDS=1
BATCH_SIZE=32
SEQ_LEN=32

echo "=== Acquiring datasets ==="
echo "---"

mkdir -p data
cd data

if [[ ! -d 'wikitext-2' ]]; then
    echo "- Downloading WikiText-2 (WT2)"
    curl https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip -o wikitext-2-v1.zip
    unzip -q wikitext-2-v1.zip
    cd wikitext-2
    mv wiki.train.tokens train.txt
    mv wiki.valid.tokens valid.txt
    mv wiki.test.tokens test.txt
    cd ..
    # TODO test this
    echo "Finished downloading data to directory /data/wikitext-2/."
    cd ..

    echo "Preprocessing text dataset..."
    echo "Default settings: batch size: 32. sequence length: 32"
    python3 build_data.py \
      --from_directory=${FROM_DIRECTORY} \
      --train_tokenizer=${TRAIN_TOKENIZER} \
      --text_file=${TEXT_DIR} \
      --vocab_size=${VOCAB_SIZE} \
      --sp_model_prefix=${SP_MODEL_PREFIX} \
      --lowercase=${LOWERCASE} \
      --shards=${SHARDS} \
      --batch_size=${BATCH_SIZE} \
      --seq_len=${SEQ_LEN} \
      --tfrecords_directory=${TFRECORD_DIR}
fi


echo " ~~~~~ Happy Training! ~~~~~ "
