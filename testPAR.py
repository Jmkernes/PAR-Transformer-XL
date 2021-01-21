import unittest
import tensorflow as tf
import numpy as np
from par_model import *
from data_utils import *

class TestPositionShifts(unittest.TestCase):

    def get_poslog_and_mask(self, qlen, max_position, d_model, direction):
        k = max_position-qlen
        q = np.random.randn(qlen, d_model)
        R = positional_encoding(max_position, d_model)[0].numpy()
        M = np.zeros((qlen, max_position))
        for i in range(qlen):
            for j in range(max_position):
                M[i,j] = q[i]@R[abs(i+k-j)]
        mask = create_lookahead_mask(qlen, max_position)
        q = tf.cast(tf.reshape(q, (1, 1, qlen, d_model)), tf.float64)
        R = tf.cast(
            tf.reshape(R, (1, 1, max_position, d_model)), tf.float64)
        R = tf.reverse(R, [2])
        pred = relative_position_logits(q, R, direction)[0,0].numpy()
        mask = mask[0,0]
        return M, pred, mask

    def test_rel_pos_logits_left(self):
        configs = [(4,4,16,'left'), (4,8,36,'left'), (1,32, 128,'left')]
        for config in configs:
            M, pred, mask = self.get_poslog_and_mask(*config)
            M *= 1-mask
            pred *= 1-mask
            self.assertTrue(np.allclose(pred, M))

    def test_rel_pos_logits_both(self):
        configs = [(4,4,16,'both'), (4,8,36,'both'), (1,32, 128,'both')]
        for config in configs:
            M, pred, mask = self.get_poslog_and_mask(*config)
            self.assertTrue(np.allclose(pred, M))

class TestMasks(unittest.TestCase):

    def test_lookahead_mask(self):
        x = create_lookahead_mask(3, 5)
        y = tf.constant([[[[0., 0., 0., 1., 1.],
         [0., 0., 0., 0., 1.],
         [0., 0., 0., 0., 0.]]]], dtype=tf.float32)
        res = tf.experimental.numpy.isclose(x, y)
        res = tf.reduce_all(res)
        self.assertTrue(res)

    def test_pad_mask(self):
        x = tf.constant([[6,7,0],[12,0,0]], dtype=tf.int64)
        x = create_pad_mask(x, 2)
        y = tf.constant(
            [[[[0., 0., 0., 0., 1.]]],
             [[[0., 0., 0., 1., 1.]]]])
        res = tf.experimental.numpy.isclose(x, y)
        res = tf.reduce_all(res)
        self.assertTrue(res)

class TestRelMultiHeadAttention(unittest.TestCase):

    def initialize(self):
        self.d_model = 64
        self.num_heads = 4
        self.max_position = 12
        self.mha = RelMultiHeadAttention(self.d_model, self.num_heads,
                                        self.max_position)

    def test_mem_is_None(self):
        if not hasattr(self, 'd_model'):
            self.initialize()
        bsz = 4
        seq_len = 6
        inp = tf.random.normal((bsz, seq_len, self.d_model))
        out = self.mha(inp, None)
        self.assertTrue( out.shape==inp.shape )

    def test_mem_not_None(self):
        if not hasattr(self, 'd_model'):
            self.initialize()
        bsz = 4
        seq_len = 6
        inp = tf.random.normal((bsz, seq_len, self.d_model))
        mem = tf.random.normal((bsz, seq_len, self.d_model))
        out = self.mha(inp, mem)
        self.assertTrue( out.shape==inp.shape )

    def test_RelaxedOneHot(self):
        roh = RelaxedOneHot()
        pi = tf.Variable([0.25,0.25,0.25,0.25])
        tau = tf.Variable([10.])
        with tf.GradientTape() as tape:
            y = roh(pi, tau)
            loss = y[0]
        grads = tape.gradient(loss, [pi, tau])
        self.assertTrue(tf.math.reduce_any(tf.greater_equal(pi, 0)))

class TestStochasticBlock(unittest.TestCase):

    def initialize(self):
        d_model, num_heads, max_position, d_ffn = 16, 2, 12, 32
        self.blk = StochasticBlock(d_model, num_heads, max_position, d_ffn)
        self.d_model = d_model


    def test_stochblock_const_tau(self):
        if not hasattr(self, 'blk'):
            self.initialize()
        bsz, seq_len = 4, 5
        tau = tf.constant(5)
        x = tf.random.uniform((bsz, seq_len, self.d_model))
        self.assertTrue( self.blk(x, None).shape==x.shape)

    def test_stochblock_variable_tau(self):
        if not hasattr(self, 'blk'):
            self.initialize()
        bsz, seq_len = 4, 5
        x = tf.random.uniform((bsz, seq_len, self.d_model))
        with tf.GradientTape() as tape:
            y = self.blk(x)
            loss = y[0,0,0]
        grad = tape.gradient(loss, [self.blk.tau])[0]
        if grad is None:
            self.assertTrue(False)
        self.assertTrue(tf.math.logical_not(tf.equal(grad, 0)))


class TestPARXL(unittest.TestCase):

    def initialize(self):
        d_model, num_heads, max_position, d_ffn = 16, 2, 12, 32
        num_layers, mem_len, vocab_size = 2, 4, 1000
        cutoffs, proj_factor = [50, 250, 1000], 2

        self.transformer = PARTransformerXL(
            d_model, num_heads, max_position,
            d_ffn, num_layers, mem_len, vocab_size,
            dropout_rate=0.1, cutoffs=cutoffs,
            proj_factor=proj_factor, proj_dims=None
        )
        self.mem_len = mem_len

        bsz, seq_len = 4, max_position-mem_len
        self.inp = tf.random.uniform(
            (bsz, seq_len), 0, vocab_size, tf.int64)
        self.labels = tf.random.uniform(
            (bsz, seq_len), 0, vocab_size, tf.int64)
        self.mem = tf.random.uniform(
            (bsz, seq_len), 0, vocab_size, tf.int64)

    def test_PARtrXL_forward_no_tau(self):
        if not hasattr(self, 'blk'):
            self.initialize()
        mems = None
        x, mems = self.transformer(self.inp, x_mems=mems)
        loss, mems = self.transformer(self.inp, x_mems=mems,
                                      labels=self.labels)

        pad_mask = create_pad_mask(self.inp, self.mem_len)
        loss, mems = self.transformer(self.inp, mems, labels=self.labels,
                                      pad_mask=pad_mask)

    def test_PARtrXL_forward_with_tau(self):
        if not hasattr(self, 'blk'):
            self.initialize()
        mems = None
        pad_mask = create_pad_mask(self.inp, self.mem_len)
        tau = tf.constant(10.)
        x, mems = self.transformer(self.inp, x_mems=mems, tau=tau)
        tau /= 2.
        loss, mems = self.transformer(self.inp, x_mems=mems, tau=tau,
                                      labels=self.labels)
        tau /= 2.
        loss, mems = self.transformer(self.inp, mems,labels=self.labels,
                                      pad_mask=pad_mask, tau=tau)

class TestDataUtils(unittest.TestCase):

    def test_find_dataset_size(self):
        data = tf.range(1000)
        ds = tf.data.Dataset.from_tensor_slices(data)
        sz = ds.reduce(0, lambda x,_: x+1)
        self.assertTrue( sz==1000 )

    def test_flatten_dataset(self):
        ds= tf.data.Dataset.from_tensor_slices([
            [b'sent', b'one'], [b'sent', b'two'], [b'sent', b'three']])
        ans = tf.constant([b'sent', b'one', b'sent', b'two',
            b'sent', b'three'], dtype=tf.string)
        pred = flatten_dataset_to_tensor(ds)
        self.assertTrue(tf.reduce_all(tf.math.equal(ans, pred)))


        ds = tf.data.Dataset.from_tensor_slices([
            b'sent', b'one', b'three'])
        ans = tf.constant([b'sent', b'one', b'three'], dtype=tf.string)
        pred = flatten_dataset_to_tensor(ds)
        self.assertTrue(tf.reduce_all(tf.math.equal(ans, pred)))

        ds = tf.data.Dataset.range(5)
        ans = tf.constant([0,1,2,3,4], dtype=tf.int64)
        pred = flatten_dataset_to_tensor(ds)
        self.assertTrue(tf.reduce_all(tf.equal(ans, pred)))

    def test_context_batch(self):
        ds_size, bsz, seq_len = 30, 5, 2
        dataset = tf.data.Dataset.range(ds_size)
        dataset, _, _ = context_batch(dataset, bsz, seq_len, ds_size)
        b1 = tf.constant([[0, 1],
                         [6, 7],
                         [12, 13],
                         [18, 19],
                         [24, 25]])
        b2 = tf.constant([[2, 3],
                         [8, 9],
                         [14, 15],
                         [20, 21],
                         [26, 27]])
        b3 = tf.constant([[4, 5],
                         [10, 11],
                         [16, 17],
                         [22, 23],
                         [28, 29]])
        b = [b1,b2,b3]
        for i, x in enumerate(dataset):
            self.assertTrue( tf.reduce_all(tf.equal(b[i], tf.cast(x, b[i].dtype))) )

    def test_serialize_token_batch(self):
        x = tf.random.uniform((64, 32), 0, 20000, tf.int32)
        proto = serialize_token_batch(x)
        y = deserialize_token_batch(proto)
        self.assertTrue( tf.reduce_all(tf.equal(x, y)) )


if __name__=="__main__":
    unittest.main()
