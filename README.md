# **P**ay **A**ttention when **R**equired (PAR) Transformer-XL
An implementation of the Pay Attention when Required transformer from the paper: https://arxiv.org/pdf/2009.04534.pdf

![alt text](https://github.com/jmkernes/PAR-Transformer-XL/blob/main/movie.gif?raw=true)
[source: Jonathan Kernes]

## Quick overview

The **P**ay **A**ttention when **R**equired Transformer (Mandava, et. al. 2020) is just a regular transformer-XL (Dai et. al. 2019)[https://arxiv.org/pdf/1901.02860.pdf]
, but the ratio of attention and dense layers has been optimized.
This optimization is performed by allowing the network to choose which types of layer it prefers in each block of the network. The present implementation is not an exact replica of the author's efforts.
Instead, we perform a simultaneous optimization procedure on both the model architecture and model parameters. The search is performed using a SuperNet, which is 
a sequential neural network composed of stochastic blocks, as shown in the figure below (taken from the paper. Please don't sue me!)

![alt text](https://github.com/jmkernes/PAR-Transformer-XL/blob/main/stoch_blks.png?raw=true)

The key component is a Gumbel-Softmax layer [(Jang et al., 2016) and (Maddison et al., 2016). jang link: https://arxiv.org/pdf/1611.01144.pdf]. This layer is a continuous representation
of a discrete sampling from a Categorical distribution, thereby allowing us to use gradients to learn parameters of a discrete distribution. 
(Recall a categorical is a distrbution over K states with kth state having probability pi_k, and we must have the normalization condition \sum_{i=1}^K pi_i = 1)

As the model learns, it is free to adjust both the usual model parameters, as well as its architecture search parameters pi, indicating the probability of choosing either

1) Attention

2) Dense

3) Identity

for any given stochastic block. We perform simulated annealing: since the categorical distribution is approximated by a continuous representation, we get some scores like (0.02, 0.98, 0.02)
for the probability of say sampling that state 2 is picked. The sharpness of this is set by a parameter \tau (the temperature), with a categorical distribution the limit tau-->0.
Simulated annealing means we begin with tau=1 to let the model figure out what it wants, then slowly decrease tau so the distribution approaches a categorical.

All of this is implemented on the freely available wiki-text2 dataset.

*Explanation of the main GIF:* The main gif is the result of our experiments. It shows the pi distribution for each stochastic block of a 6 block SuperNet, as a function of training iterations.
The number indicates the probability of the most likely layer type (darker means more probable). As you can see, the model learns to put attention in the beginning, and dense layers at the end.


## Data

The dataset used is Wiki-text2. We have provided a copy of this in the data folder, along with some preprocessed data for training. In order to reproduce this from scratch, run the shell script

```
./create_tfrecords.sh
```

This will download the wiki-text2 dataset from its source, then proceed to clean, batch, and write the data to a tfrecords file. The shell script calls ```build_data.py``` which offers more control over what type of data to generate. The general parameters you will want to tune are:

*batch_size 
*seq_len.

You can also supply your own dataset instead of the one provided. The underlying tokenizer uses sentencepiece (Kudo): https://github.com/google/sentencepiece, which works at the byte level and can handle any kind of input. Simply change the --input_text flag to your file, and set the desired --vocab_size.

Why do we need to specify the batch size? Transformer XL uses memory states to form a recurrent, long range network. After analyzing a particular sequence say [A,B] of the sequence [A,B,C,D], the results of [A,B] are fed into the [C,D] calculation with a stop gradient. Therefore, we must be sure that each datapoint follows chronologically from the previous one.

This is achieved by *context batching* (see data_utils.py function) where we break the entire dataset into batch_size segments, then pull in order one sequence from each batch at a time to form the dataset. Because of this, note that adding more shards to the data could result in a large loss (order of batch_size\*seq_len\*shards), as each shard will drop the remaining datapoint of size (batch_size\*seq_len) to keep the tensor shapes.


## Addtional technical details

Per the original Transformer-XL, we also implement an adaptive softmax layer (Grave et. al. 2017, https://arxiv.org/abs/1609.04309) to deal with a potentially large number of outputs in the final dense layer. This implemenation is inspired by the TF 1.0 example at https://github.com/yangsaiyong/tf-adaptive-softmax-lstm-lm.
To use the adaptive softmax, set the ```--cutoffs=``` flag in train.py. The cutoffs are the max values of each bin, and should NOT include the vocab size (i.e. the max cutoff of the final bin). If no cutoffs are specified, the model defaults to normal softmax.

For completeness, we have also provided a script ```optimal_cuts.py``` that determines the optimal cutoffs given a return space separated file of unigram probabilities (based on the assumptions of Grave et. al. regarding GPU computation complexity -- see the paper for details). 
The algorithm uses dynamic programming, but is quite slow at O(KN^2), for K cutoffs and N vocab words. In principle it's a one time cost to determine the cutoffs, but we are impatient and recommend to just play around with the cutoffs instead. See the script for flag details

## Training and Benchmarks

The default model we use has memory length 16, feed-forward dimension 1024, attention dimension 128, and 6 stochastic blocks, with an adaptive softmax layer and 2 clusters. We trained on a colab GPU for 20 epochs, taking a total of 37 minutes. We use an Adam optimzer with cosine rate decay: an initial warmup of 4000 steps and a maximum learning rate of 1e-4, decaying to zero at the end of training. Our training benchmarks are:

| Iteration (thousands) | Train_perplexity | Validation_perplexity | Time    |
|:---------------------:|:----------------:|-----------------------|---------|
|          2.7k         |       163.9      |         114.4         |  1m 58s |
|          8.5k         |       78.56      |         62.33         |  5m 37s |
|         14.1k         |       65.71      |         51.88         |  9m 28s |
|         28.3k         |       48.52      |         42.61         | 18m 40s |
|         48.1k         |       41.85      |         39.57         | 31m 51s |
|         56.5k         |       42.12      |         39.41         | 37m 14s |


To train, simply run the shell script
```
./base_model.sh
```
adjusting the parameters as you see fit. The above model is the default configuration. To train in colab, simply open up the notebook "colab.ipynb" and follow the instructions. This is most easily done by going to [google.colab.com] and searching this repository in github. The benefit of colab, is it's easier to play around with the model after training.

While training, we have provided two ways to monitor the output

1) A tensorboard log. The colab notebook takes care of running this for you. In the terminal, first create a 'logs' directory, then run the command ```tensorboard --logdir logs``` in a separate tab. This will open a port where you can view live plots of the learning rate, tau annealing, train/valid loss and perplexity.

2) An output log saved to training_log.log. This will log the model summary, parameters, etc. as well as print out loss updates every 100 steps and save it to the log file.

## Thanks for reading this far!

Enjoy! And thank you to the wonderful researchers that inspired this project.

If you would like to contribute, or have any comments questions concerns please open a pull request or email me directly.
