import tensorflow as tf
from adaptive_softmax import AdaptiveSoftmax
from par_model import create_pad_mask, create_lookahead_mask, positional_encoding, positionwise_ffn, RelaxedOneHot

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert not d_model%num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.Wq = tf.keras.layers.Dense(d_model)
        self.Wk = tf.keras.layers.Dense(d_model)
        self.Wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model, activation='relu')

    def split_heads(self, x):
        x = tf.reshape(x, (x.shape[0], x.shape[1], -1, self.d_model//self.num_heads))
        return tf.transpose(x, [0,2,1,3])

    def call(self, q, k, v, mask=None):
        out_shape = q.shape
        Q = self.split_heads(self.Wq(q))
        K = self.split_heads(self.Wk(k))
        V = self.split_heads(self.Wv(v))
        scale = tf.math.rsqrt(tf.cast(self.d_model, tf.float32))
        scaled_logits = tf.matmul(Q, K, transpose_b=True)*scale
        if mask is not None:
            scaled_logits += -1e9*mask
        weights = tf.nn.softmax(scaled_logits, axis=-1)
        attn = tf.transpose(tf.matmul(weights, V), [0,2,1,3])
        attn = tf.reshape(attn, out_shape)
        return self.dense(attn)

class VanillaStochasticBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ffn, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = positionwise_ffn(d_model, d_ffn)
        self.gumbel_softmax = RelaxedOneHot()
        self.layernorm_mha = tf.keras.layers.LayerNormalization()
        self.layernorm_ffn = tf.keras.layers.LayerNormalization()
        self.dropout_mha = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)
        self.pi = tf.Variable([0.33,0.33,0.34], name='pi')
        self.tau = tf.Variable(1., name='tau')

    def call(self, x, tau=None, training=None, mask=None):
        if tau is None:
            tau = self.tau # if no schedule, use the variable
        block1 = self.dropout_mha(self.mha(x, x, x, mask), training=training)
        block1 = self.layernorm_mha(block1+x)
        block2 = self.dropout_ffn(self.ffn(x), training=training)
        block2 = self.layernorm_ffn(block2+x)
        weights = self.gumbel_softmax(self.pi, tau)
        output = block1*weights[0] + block2*weights[1] + x*weights[2]
        return output

class VanillaPARtransformer(tf.keras.Model):
    def __init__(self, d_model, num_heads, d_ffn, num_layers, vocab_size, max_position,
     dropout_rate=0.1, cutoffs=None, proj_factor=4, proj_dims=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.cutoffs = cutoffs
        self.embed = tf.keras.layers.Embedding(vocab_size, d_model)
        self.final_layer = tf.keras.layers.Dense(vocab_size) if cutoffs is None else AdaptiveSoftmax(cutoffs, proj_factor, proj_dims)
        self.stoch_blks = [VanillaStochasticBlock(d_model, num_heads, d_ffn, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name='inp_dropout')

    def _loss(self, hidden_state, labels):
        if self.cutoffs is not None:
            return self.final_layer(hidden_state, labels)
        logits = self.final_layer(hidden_state)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        return tf.reduce_mean(loss)

    def call(self, x, position, tau=None, labels=None, training=None, mask=None):
        x = self.embed(x) + position
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.stoch_blks[i](x, tau, training, mask)
        return self.final_layer(x) if labels is None else self._loss(x, labels)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ffn, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = positionwise_ffn(d_model, d_ffn)
        self.layernorm_mha = tf.keras.layers.LayerNormalization()
        self.layernorm_ffn = tf.keras.layers.LayerNormalization()
        self.dropout_mha = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=None, mask=None):
        out1 = self.dropout_mha(self.mha(x, x, x, mask), training=training)
        out1 = self.layernorm_mha(out1+x)
        out2 = self.dropout_ffn(self.ffn(out1), training=training)
        out2 = self.layernorm_ffn(out2+out1)
        return out2

class VanillaTransformer(tf.keras.Model):
    def __init__(self, d_model, num_heads, d_ffn, num_layers, vocab_size, max_position,
     dropout_rate=0.1, cutoffs=None, proj_factor=4, proj_dims=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.cutoffs = cutoffs
        self.embed = tf.keras.layers.Embedding(vocab_size, d_model)
        self.encoder = tf.keras.models.Sequential([
            EncoderLayer(d_model, num_heads, d_ffn, dropout_rate) for _ in range(num_layers)
        ])
        if cutoffs is None:
            self.final_layer = tf.keras.layers.Dense(vocab_size)
        else:
            self.final_layer = AdaptiveSoftmax(cutoffs, proj_factor, proj_dims)
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name='inp_dropout')

    def _loss(self, hidden_state, labels):
        if self.cutoffs is not None:
            return self.final_layer(hidden_state, labels)
        logits = self.final_layer(hidden_state)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        return tf.reduce_mean(loss)

    def call(self, x, position, tau=None, labels=None, training=None, mask=None):
        x = self.embed(x) + position
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.dropout(x, training=training)
        x = self.encoder(x)
        return self.final_layer(x) if labels is None else self._loss(x, labels)

# self.pos_enc = positional_encoding(max_position, d_model)
# self.lookahead_mask = create_lookahead_mask(max_position, max_position)
