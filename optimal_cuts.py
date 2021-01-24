from absl import app
from absl import flags
import numpy as np
import collections
import time

FLAGS = flags.FLAGS

flags.DEFINE_string('filename', '', 'Return spaced file of vocab counts. Need not be sorted')
flags.DEFINE_integer('clusters', 1, 'Desired number of clusters.', lower_bound=1)
flags.DEFINE_boolean('verbose', True, 'Displays training progress.')

def load_vocab(filename):
    vocab = []
    with open(filename, 'r') as file:
        for line in file:
            p = float(line.strip())
            vocab.append(p)
    return vocab


def get_partitions(sequence, num_clusters, verbose):
    tot_start = time.time()
    sequence = sorted(sequence, reverse=True)
    seq_len = len(sequence)
    num_cuts = num_clusters-1

    # prefix sum for quick summations
    F = [0]
    for x in sequence:
        F.append(F[-1]+x)

    # initialize the dp array and optimal cut dict
    d = np.full((num_cuts+1, seq_len), np.inf)
    d[0] = np.array(F[1:])*np.arange(1,seq_len+1)
    c = collections.defaultdict(list)

    # for each new cut, the optimal partition is given by curr
    # find all optimal partitions of seqs of length 0->seq_len
    # in order, and store these in d for quick recall.
    for cut in range(num_cuts):
        if verbose:
            print("Cutting cluster:",cut+1)
        start = time.time()
        for j in range(seq_len):
            for i in range(cut,j):
                curr = d[cut,i]+(j-i)*(F[j+1]-F[i+1])
                if curr <= d[cut+1, j]:
                    d[cut+1,j] = curr
                    c[cut,j] = c[cut-1,i]+[i]
            if verbose:
                end = '\r' if j<seq_len-1 else '\n'
                print(f"{100*(j+1)/seq_len}% complete, {time.time()-start:.0f}s elapsed.", end=end)
        if verbose:
            print(f"{time.time()-start:.2f}s for solving {cut+1} cluster(s).")
    cost = d[num_cuts, seq_len-1]
    cuts = c[num_cuts-1, seq_len-1]
    if verbose:
        t = time.time()-tot_start
        minutes = int(t)//60
        seconds = int(t)%60
        if not seconds and not minutes:
            print(f'~~~ Finished. Total time: {t:.2f}s ~~~')
        else:
            print(f'~~~ Finished. Total time: {minutes:0d}m {seconds:0d}s ~~~')
    return cuts

def main(argv):
    if not FLAGS.filename:
        print("Invalid filename.")
        return
    filename = FLAGS.filename
    verbose = FLAGS.verbose
    clusters = FLAGS.clusters

    vocab = load_vocab(filename)
    vocab_size = len(vocab)

    if verbose:
        print('-'*10,f' Finding optimal cuts for file {filename} ', '-'*10)
        print(f'Vocabulary size: {vocab_size}')
    cutoffs = get_partitions(vocab, clusters, verbose)

    cutoffs = [i-1 for i in cutoffs]
    cutoffs.append(vocab_size)
    print("The optimal cuts are", cutoffs)
    return cutoffs

if __name__ == '__main__':
  app.run(main)
