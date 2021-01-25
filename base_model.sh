#!/bin/sh

echo "=== Setting up configuration ==="

# Data stuff
TRAIN_DIR=data/wikitext2_bsz32_seqlen32_tfrecords_train
VAL_DIR=data/wikitext2_bsz32_seqlen32_tfrecords_valid
TEST_DIR=data/wikitext2_bsz32_seqlen32_tfrecords_test
SP_MODEL_PREFIX=tokenizer/wiki2_12k

# Model stuff
D_MODEL=128
NUM_HEADS=4
D_FFN=512
NUM_LAYERS=12
MEM_LEN=32
DROPOUT_RATE=0.1
CUTOFF1=250
CUTOFF2=2500
PROJ_FACTOR=4
STRAIGHT_THROUGH=False

# Don't set proj_dims
MAX_LR=1e-4
WARMUP_STEPS=4000
TAU_START=1.0
TAU_END=0.1
EPOCHS=20
TAU_IS_TRAINABLE=False
OPT_NAME=adam

# File prefix for checkpointing and TensorBoard
MODEL_NAME=dmodel256_dffn1024_blocks12

echo "=== Beginning training ==="
python3 train.py \
  --train_directory=${TRAIN_DIR} \
  --valid_directory=${VAL_DIR} \
  --test_directory=${TEST_DIR} \
  --sp_model_prefix=${SP_MODEL_PREFIX} \
  --d_model=${D_MODEL} \
  --num_heads=${NUM_HEADS} \
  --d_ffn=${D_FFN} \
  --num_layers=${NUM_LAYERS} \
  --mem_len=${MEM_LEN} \
  --dropout_rate=${DROPOUT_RATE} \
  --cutoffs=${CUTOFF1} \
  --cutoffs=${CUTOFF2} \
  --proj_factor=${PROJ_FACTOR} \
  --warmup_steps=${WARMUP_STEPS} \
  --tau_start=${TAU_START} \
  --tau_end=${TAU_END} \
  --tau_is_trainable=${TAU_IS_TRAINABLE} \
  --opt_name=${OPT_NAME} \
  --epochs=${EPOCHS} \
  --model_name=${MODEL_NAME} \
  --max_lr=${MAX_LR} \
  --straight_through=${STRAIGHT_THROUGH}
