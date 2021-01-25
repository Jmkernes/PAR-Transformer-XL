import tensorflow as tf
from adaptive_softmax import AdaptiveSoftmax


def positional_encoding(max_position, d_embed, min_freq=1e-4):
    position = tf.range(max_position, dtype=tf.float32)
    mask = tf.range(d_embed)
    sin_mask = tf.cast(mask%2, tf.float32)
    cos_mask = 1-sin_mask
    exponent = 2*(mask//2)
    exponent = tf.cast(exponent, tf.float32)/tf.cast(d_embed, tf.float32)
    freqs = min_freq**exponent
    angles = tf.einsum('i,j->ij', position, freqs)
    pos_enc = tf.math.cos(angles)*cos_mask + tf.math.sin(angles)*sin_mask
    return pos_enc[tf.newaxis, :, :]

def create_lookahead_mask(query_len, key_len, dtype=tf.float32):
    mask = 1-tf.linalg.band_part(
        tf.ones((key_len, key_len), dtype=dtype), -1, 0)
    mask = mask[tf.newaxis, tf.newaxis, -query_len:, :]
    return mask

def create_pad_mask(x, mem_len):
    """Assumes input of form x: (N, qlen). Masks keys in the attention logit
    matrix corresponding to positions of pad element in input. Needs mem_len.
    Only needed when processing final window of an input sequence."""
    x = tf.cast(tf.equal(x,0), dtype=tf.float32)
    x = tf.pad(x, [(0,0),(mem_len,0)])
    return x[:, tf.newaxis, tf.newaxis, :]

def left_shift(x):
    dims = tf.shape(x)
    x = tf.pad(x, [(0,0),(0,0),(0,0),(1,0)])
    x = tf.reshape(x, (dims[0], dims[1], dims[3]+1, dims[2]))
    x = x[:,:,1:,:]
    return tf.reshape(x, dims)

def right_shift(x):
    dims = x.shape
    x = tf.pad(x, [(0,0),(0,0),(0,0),(0,1)])
    x = tf.reshape(x, (dims[0], dims[1], dims[3]+1, dims[2]))
    x = x[:,:,:-1,:]
    return tf.reshape(x, dims)

def relative_position_logits(q, R, direction='left'):
    """ For efficient, assumes that R has been flipped along the
    position axis, i.e. R -> tf.reverse(R, [2])"""
    if direction=='left':
        return left_shift(tf.matmul(q, R, transpose_b=True))
    elif direction=='right':
        R_flipud = tf.reverse(R, [2])
        return right_shift(tf.matmul(q, R_flipud, transpose_b=True))
    elif direction=='both':
        R_flipud = tf.reverse(R, [2])
        num_to_roll = R.shape[2]-q.shape[2]
        lower = left_shift(tf.matmul(q, R, transpose_b=True))
        upper = right_shift(tf.matmul(q, R_flipud, transpose_b=True))
        upper = tf.roll(upper, shift=num_to_roll, axis=3)
        mask = create_lookahead_mask(q.shape[2], R.shape[2], lower.dtype)
        return (1-mask)*lower + (mask)*upper
    else:
        raise ValueError("Choose valid direction.")

class RelMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, max_position, **kwargs):
        super().__init__(**kwargs)
        assert not d_model%num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model//num_heads
        self.max_position = max_position

        self.R = positional_encoding(max_position, d_model)
        self.R = tf.reverse(self.R, [2])

        self.W_emb = tf.keras.layers.Dense(3*d_model)
        self.W_pos = tf.keras.layers.Dense(d_model)

        self.u = self.add_weight(
            name="global_content_bias",
            shape=[1, num_heads, 1, self.depth],
            initializer="glorot_normal"
        )

        self.v = self.add_weight(
            name="global_position_bias",
            shape=[1, num_heads, 1, self.depth],
            initializer="glorot_normal"
        )

        self.dense = tf.keras.layers.Dense(d_model, activation='relu')

    def split_heads(self, x):
        x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], -1, self.depth))
        return tf.transpose(x, [0,2,1,3])

    def call(self, x, x_mem, pad_mask=None):
        # T_q=query_len, T_k=key_len, T_k=T_q+T_mem
        batch_size = tf.shape(x)[0]
        query_len  = tf.shape(x)[1]

        # Extract query, key, and values
        if x_mem is not None:
            x = tf.concat([x_mem, x], 1) # (N,T_k,D) concat along T

        # (R contains max position ever needed. If not needed, truncate)
        key_len = tf.shape(x)[1]
        R = self.R[:,:key_len,:] # only does something when x_mem is None

        QVK = self.W_emb(x) # qvk is (N,T_k,3*D)
        R = self.W_pos(R) # R: (1,T_k,D) -> (1,T_k,D)
        Q,V,K = tf.split(QVK, 3, -1) # q,v,k are all (N,T_k,D)
        Q = self.split_heads(Q[:,-query_len:,:]) # (N,heads,T_q,depth)
        V = self.split_heads(V) # (N,heads,T_k,depth)
        K = self.split_heads(K) # (N,heads,T_k,depth)
        R = self.split_heads(R) # (1,heads,T_k,depth)

        # Form logits. These are (N, heads, T_q, T_k)
        AC = tf.matmul(Q+self.u, K, transpose_b=True) # embedding logits
        BD = relative_position_logits(Q+self.v, R) # position logits
        scale = tf.cast(tf.shape(K)[-1], tf.float32) # sqrt(d_model)
        scaled_logits = (AC+BD)/scale

        # Masking. create ones on upper diag of (T_k,T_k) then truncate
        # ensures queries see first mem_len keys, then starts masking
        # pad_mask should act the same way.
        mask = create_lookahead_mask(query_len, key_len)
        if pad_mask is not None:
            mask += pad_mask

        # Attention.
        scaled_logits += -1e9*mask
        weights = tf.nn.softmax(scaled_logits, -1) # (N,heads,T_q)
        attn = tf.matmul(weights, V) # (N,heads,T_q,depth)
        attn = tf.transpose(attn, [0,2,1,3])
        attn = tf.reshape(attn, (batch_size, query_len, -1))

        output = self.dense(attn)
        return output

def positionwise_ffn(d_model, d_fnn, activation='relu'):
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(d_fnn, activation=activation),
        tf.keras.layers.Dense(d_model)
    ])

class RelaxedOneHot(tf.keras.layers.Layer):
    def __init__(self, straight_through=False, **kwargs):
        super().__init__(**kwargs)
        self.straight_through = straight_through

    def build(self, input_shape):
        self.K = input_shape[-1] # K categories, i.e. dim(pi)
        super().build(input_shape)

    def gumbel(self, size=(1,)):
        x = tf.random.uniform(size)
        return -tf.math.log(-tf.math.log(x))

    def call(self, pi, tau):
        eps = 1e-5 # for numerical stability. hopefully...
        g = self.gumbel(tf.shape(pi))
        logits = (tf.math.log(pi+eps)+g)/tau
        out = tf.nn.softmax(logits)
        if self.straight_through:
            out_hard = tf.one_hot(tf.argmax(out, -1), self.K)
            out = tf.stop_gradient(out_hard-out)+out
        return out

class StochasticBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, max_position, d_ffn,
                dropout_rate=0.1, straight_through=False, **kwargs):
        super().__init__(**kwargs)
        self.rmha = RelMultiHeadAttention(d_model, num_heads, max_position)
        self.ffn = positionwise_ffn(d_model, d_ffn)
        self.gumbel_softmax = RelaxedOneHot(straight_through)

        self.layernorm_mha = tf.keras.layers.LayerNormalization()
        self.layernorm_ffn = tf.keras.layers.LayerNormalization()

        self.dropout_mha = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

        # I tried a uniform initialization at first, but it doesn't train as well
        self.pi = tf.Variable([0.33,0.33,0.34], name='pi')
#        pi_init = tf.random.uniform((3,), dtype=tf.float32)
#        pi_init = pi_init/tf.reduce_sum(pi_init)
#        self.pi = tf.Variable(pi_init)
        # Empirically, letting the model learn tau causes NaN issues.
        self.tau = tf.Variable(1., name='tau')

    def call(self, x, x_mem=None, tau=None, training=None, pad_mask=None):
        # NOTE: I think their layernorm in the paper is a mistake.
        # I think it should be how I have it.
        if tau is None:
            tau = self.tau # if no schedule, use the variable

        # During training, we need to ensure pi is positive and sums to 1.
        if training:
            self.pi = tf.maximum(self.pi, 0)
            self.pi = self.pi/tf.reduce_sum(self.pi)

        # mha block
        block1 = self.rmha(x, x_mem, pad_mask)
        block1 = self.dropout_mha(block1, training=training)
        block1 = self.layernorm_mha(block1+x)

        # ffn block
        block2 = self.ffn(x)
        block2 = self.dropout_ffn(block2, training=training)
        block2 = self.layernorm_ffn(block2+x)

        # Gumbel softmax
        weights = self.gumbel_softmax(self.pi, tau, training=training)

        output = block1*weights[0] + block2*weights[1] + x*weights[2]
        return output


from adaptive_softmax import AdaptiveSoftmax

class PARTransformerXL(tf.keras.Model):
    def __init__(self, d_model, num_heads, max_position,
                 d_ffn, num_layers, mem_len, vocab_size,
                 dropout_rate=0.1, cutoffs=None, proj_factor=4,
                 proj_dims=None, straight_through=False, **kwargs):
        super().__init__(**kwargs)
        assert mem_len >= 0 and max_position > 0

        self.d_model = d_model
        self.mem_len = mem_len
        self.cutoffs = cutoffs
        self.num_layers = num_layers
        self.max_position = max_position

        self.embed = tf.keras.layers.Embedding(vocab_size, d_model)

        if cutoffs:
            self.final_layer = AdaptiveSoftmax(cutoffs, proj_factor, proj_dims)
        else:
            self.final_layer = tf.keras.layers.Dense(vocab_size)

        self.stoch_blks = [
            StochasticBlock(d_model, num_heads, max_position, d_ffn,
                            dropout_rate, straight_through)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate, name='inp_dropout')

    def _get_next_mem(self, x, x_mem, mem_len):
        if mem_len==0:
            return x_mem # ensures None is always returned. May break in graph mode!!!
        elif x_mem is None:
            return tf.stop_gradient(x[:,-mem_len:,:])
        new_state = tf.concat([x_mem, x], 1)
        new_state = new_state[:,-mem_len:,:]
        return tf.stop_gradient(new_state) # don't backpropagate to cache

    def _loss(self, hidden_state, labels):
        if self.cutoffs:
            return self.final_layer(hidden_state, labels)
        logits = self.final_layer(hidden_state)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        return tf.reduce_mean(loss)

    def call(self, x, x_mems=None, labels=None, tau=None, training=None, pad_mask=None):

        if x_mems is None:
            x_mems = [None]*self.num_layers

        x = self.embed(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.dropout(x, training=training)

        new_mems = []
        for i in range(self.num_layers):
            new_mem = self._get_next_mem(x, x_mems[i], self.mem_len)
            new_mems.append(new_mem)
            x = self.stoch_blks[i](x, x_mems[i], tau, training, pad_mask)

        if labels is not None:
            loss = self._loss(x, labels)
            return loss, new_mems

        probs = self.final_layer(x)
        return probs, new_mems
