import tensorflow as tf

class AdaptiveSoftmax(tf.keras.layers.Layer):
    """ An adaptive softmax layer suited for large output dimension. and GPUs
    Based on https://arxiv.org/pdf/1609.04309.pdf, and on the implementation
    at https://github.com/yangsaiyong/tf-adaptive-softmax-lstm-lm"""
    def __init__(self, cutoffs, proj_factor=4, proj_dims=[], **kwargs):
        """ Vocabulary must be sorted in decreasing order of frequency.
        Args:
            cutoffs: a list of integers giving bin cutoffs. The last element
                    must be the vocab_size. Bins are intervals like [i,j).
            proj_factor: reduction factor for hidden states feeding into
                            successive clusters. Example- let h0=512 and
                            project_factor=2. Then cluster1 will have
                            h1=256, cluster2 has h2=128, etc.
            proj_dims (default=None): if specified, overrides project_factor
        Methods:
            loss: takes inputs and targets. returns loss
            softmax: returns (batch, seq_len, vocab_size) tensor of probs
            log_softmax: same as softmax, but log probabilities
            call: if target is input, returns loss, else returns softmax
        """
        super().__init__(**kwargs)
        assert cutoffs # Made these two first additions. please don't break everything!
        self.cutoffs = sorted(cutoffs)
        self.cluster_num = len(cutoffs)-1
        self.proj_dims = proj_dims
        self.proj_factor = proj_factor
        self.head_w = None
        self.tail_w = None

    def get_config(self):
        proj_dims = [x.numpy() for x in self.proj_dims]
        base_config = super(AdaptiveSoftmax, self).get_config()
        return {**base_config, "proj_factor": self.proj_factor,
         "proj_dims":proj_dims, "cutoffs":self.cutoffs}

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        if self.proj_dims:
            assert len(self.proj_dims)==self.cluster_num
        else:
            self.proj_dims = [tf.math.maximum(hidden_dim/self.proj_factor**i, 1)
            for i in range(1, self.cluster_num+1)]

        # initialize dense layers for head. reserve cluster_num spots for
        # each cluster probability
        head_dim = self.cutoffs[0]+self.cluster_num
        self.head_w = tf.keras.layers.Dense(head_dim, name='ada_softmax_head_w',input_shape=[None, hidden_dim])

        # initialize dense layers for each cluster.
        self.tail_w = []
        for i in range(self.cluster_num):
            tail_dim = self.cutoffs[i+1]-self.cutoffs[i]
            self.tail_w.append(
                tf.keras.models.Sequential([
                    tf.keras.layers.Dense(self.proj_dims[i], name=f"ada_softmax_tail{i+1}_proj", input_shape=[None, hidden_dim]),
                    tf.keras.layers.Dense(tail_dim, name=f"ada_softmax_tail{i+1}_w")
                ])
            )
        super().build(input_shape)

    def _loss(self, inp, labels, reduction='auto'):
        head_labels = labels
        loss = tf.sparse.SparseTensor([[0,0]],[0.], tf.cast(tf.shape(labels), tf.int64))
        for i in range(self.cluster_num):
            mask = tf.logical_and(
                tf.greater_equal(labels, self.cutoffs[i]),
                tf.less(labels, self.cutoffs[i+1])
            )
            if tf.logical_not(tf.math.reduce_any(mask)):
                continue

            # separate the cluster prob from label_prob to prevent underflow
            # i.e. calculate softmax-loss on cluster prob independently of
            # labels within that cluster
            cluster_label = tf.cast(self.cutoffs[0]+i, head_labels.dtype)
            head_labels = tf.where(mask, cluster_label, head_labels)

            # NOTE: taking boolean mask will help with efficiency, which is the whole reason
            # that we need a built-in loss (to only calculate softmax for the bin that the label tells us to)
            # BUT it will collapse a dimension, i.e. it maps (bsz, seq_len, d_model) to
            # (num_true, d_model) this is a different input shape to the dense layer than when we call the
            # logits function. Thus, we must pad the zero dimension to be consistent.
            tail_inp = tf.boolean_mask(inp, mask) # (num_in_clust, hidden_dim)
            tail_logits = self.tail_w[i](tf.expand_dims(tail_inp,0))[0] # expand then contract back
            tail_labels = tf.boolean_mask(labels-self.cutoffs[i], mask)
            tail_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                tail_labels, tail_logits)
            aligned_tail_loss = tf.sparse.SparseTensor(
                indices=tf.where(mask), values=tail_loss,
                dense_shape=tf.cast(tf.shape(labels), tf.int64)
            )

            # don't convert these to dense yet. slightly more efficient.
            loss = tf.sparse.add(loss, aligned_tail_loss)

        head_logits = self.head_w(inp)
        head_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    head_labels, head_logits)
        loss = head_loss + tf.sparse.to_dense(loss)

        # Input labels 0, are assumed to represent padding
        # and must be masked from the loss
        pad_mask = tf.math.logical_not(tf.math.equal(labels, 0))
        pad_mask = tf.cast(pad_mask, loss.dtype)
        loss *= pad_mask

        return tf.reduce_mean(loss) if reduction=='auto' else loss

    def _softmax(self, inp):
        head_logits = self.head_w(inp)
        head_softmax = tf.nn.softmax(head_logits)
        output = head_softmax[:, :, :-self.cluster_num]
        for i in range(self.cluster_num):
            tail_logits = self.tail_w[i](inp)
            tail_softmax = tf.nn.softmax(tail_logits)
            cluster_id = self.cutoffs[0]+i
            tail_softmax *= head_softmax[:, :, cluster_id:cluster_id+1]
            output = tf.concat([output, tail_softmax], axis=-1)
        return output

    def _log_softmax(self, inp):
        head_logits = self.head_w(inp)
        head_softmax = tf.nn.log_softmax(head_logits)
        output = head_softmax[:, :, :-self.cluster_num]
        for i in range(self.cluster_num):
            tail_logits = self.tail_w[i](inp)
            tail_softmax = tf.nn.log_softmax(tail_logits)
            cluster_id = self.cutoffs[0]+i
            tail_softmax += head_softmax[:, :, cluster_id:cluster_id+1]
            output = tf.concat([output, tail_softmax], axis=-1)
        return output

    def call(self, inp, labels=None, use_log=False, reduction='auto'):
        if labels is None:
            return self._log_softmax(inp) if use_log else self._softmax(inp)
        return self._loss(inp, labels, reduction)
