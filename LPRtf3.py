import tensorflow as tf
import numpy as np
import time
import cv2
import os
import random

#训练最大轮次
num_epochs = 300

#初始化学习速率
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 2000
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
MOMENTUM = 0.9

#输出字符串结果的步长间隔
REPORT_STEPS = 5000

#训练集的数量
BATCH_SIZE = 50
TRAIN_SIZE = 7368
BATCHES = TRAIN_SIZE//BATCH_SIZE
test_num = 3

ti = 'train'         #训练集位置
vi = 'valid'         #验证集位置
img_size = [94, 24]
tl = None
vl = None
num_channels = 3
label_len = 7

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z','_'
         ]
dict = {'A01':'京','A02':'津','A03':'沪','B02':'蒙',
        'S01':'皖','S02':'闽','S03':'粤','S04':'甘',
        'S05': '贵', 'S06': '鄂', 'S07': '冀', 'S08': '黑', 'S09': '湘',
        'S10': '豫', 'S12': '吉', 'S13': '苏', 'S14': '赣', 'S15': '辽',
        'S17': '川', 'S18': '鲁', 'S22': '浙',
        'S30':'渝', 'S31':'晋', 'S32':'桂', 'S33':'琼', 'S34':'云', 'S35':'藏',
        'S36':'陕','S37':'青', 'S38':'宁', 'S39':'新'}

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

NUM_CHARS = len(CHARS)


def encode_label(s):
    label = np.zeros([len(s)])
    for i, c in enumerate(s):
        label[i] = CHARS_DICT[c]
    return label

#读取图片和label,产生batch
class TextImageGenerator:
    def __init__(self, img_dir, label_file, batch_size, img_size, num_channels=3, label_len=7):
        self._img_dir = img_dir
        self._label_file = label_file
        self._batch_size = batch_size
        self._num_channels = num_channels
        self._label_len = label_len
        self._img_w, self._img_h = img_size

        self._num_examples = 0
        self._next_index = 0
        self._num_epoches = 0
        self.filenames = []
        self.labels = []

        self.init()

    def init(self):
        self.labels = []
        fs = os.listdir(self._img_dir)
        for filename in fs:
                self.filenames.append(filename)
        for filename in self.filenames:
            if '\u4e00' <= filename[0]<= '\u9fff':
                label = filename[:7]
            else:
                label = dict[filename[:3]] + filename[4:10]
            label = encode_label(label)
            self.labels.append(label)
            self._num_examples += 1
        self.labels = np.float32(self.labels)

    def next_batch(self):
        # Shuffle the data
        if self._next_index == 0:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._filenames = [self.filenames[i] for i in perm]
            self._labels = self.labels[perm]

        batch_size = self._batch_size
        start = self._next_index
        end = self._next_index + batch_size
        if end > self._num_examples:
            self._next_index = 0
            start = self._next_index
            end = self._next_index + batch_size
            self._num_epoches += 1
        else:
            self._next_index = end
        images = np.zeros([batch_size, self._img_h, self._img_w, self._num_channels])
        # labels = np.zeros([batch_size, self._label_len])

        for j, i in enumerate(range(start, end)):
            fname = self._filenames[i]
            img = cv2.imread(os.path.join(self._img_dir, fname))
            img = cv2.resize(img, (self._img_w, self._img_h), interpolation=cv2.INTER_CUBIC)
            images[j, ...] = img
        images = np.transpose(images, axes=[0, 2, 1, 3])
        labels = self._labels[start:end, ...]
        targets = [np.asarray(i) for i in labels]
        sparse_labels = sparse_tuple_from(targets)
        # input_length = np.zeros([batch_size, 1])

        seq_len = np.ones(self._batch_size) * 24
        return images, sparse_labels, seq_len

def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result


def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = CHARS[spars_tensor[1][m]]
        decoded.append(str)
    return decoded

def small_basic_block(x,im,om):
    x = conv(x,im,int(om/4),ksize=[1,1])
    x = tf.nn.relu(x)
    x = conv(x,int(om/4),int(om/4),ksize=[3,1],pad='SAME')
    x = tf.nn.relu(x)
    x = conv(x,int(om/4),int(om/4),ksize=[1,3],pad='SAME')
    x = tf.nn.relu(x)
    x = conv(x,int(om/4),om,ksize=[1,1])
    return x

def conv(x,im,om,ksize,stride=[1,1,1,1],pad = 'SAME'):
    conv_weights = tf.Variable(
        tf.truncated_normal([ksize[0], ksize[1], im, om],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=None, dtype=tf.float32))
    conv_biases = tf.Variable(tf.zeros([om], dtype=tf.float32))
    out = tf.nn.conv2d(x,
                        conv_weights,
                        strides=stride,
                        padding=pad)
    relu = tf.nn.bias_add(out, conv_biases)
    return relu

def get_train_model(num_channels, label_len, b, img_size):
    inputs = tf.placeholder(
        tf.float32,
        shape=(b, img_size[0], img_size[1], num_channels))

    # 定义ctc_loss需要的稀疏矩阵
    targets = tf.sparse_placeholder(tf.int32)

    # 1维向量 序列长度 [batch_size,]
    seq_len = tf.placeholder(tf.int32, [None])
    x = inputs

    x = conv(x,num_channels,64,ksize=[3,3])
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x,
                          ksize=[1, 3, 3, 1],
                          strides=[1, 1, 1, 1],
                          padding='SAME')
    x = small_basic_block(x,64,64)
    x2=x
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.nn.max_pool(x,
                          ksize=[1, 3, 3, 1],
                          strides=[1, 2, 1, 1],
                          padding='SAME')
    x = small_basic_block(x, 64,256)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = small_basic_block(x, 256, 256)
    x3 = x
    x = tf.layers.batch_normalization(x)

    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 1, 1],
                       padding='SAME')
    x = tf.layers.dropout(x)

    x = conv(x, 256, 256, ksize=[4, 1])
    x = tf.layers.dropout(x)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)


    x = conv(x,256,NUM_CHARS+1,ksize=[1,13],pad='SAME')
    x = tf.nn.relu(x)
    cx = tf.reduce_mean(tf.square(x))
    x = tf.div(x,cx)

    #x = tf.reduce_mean(x,axis = 2)
    #x1 = conv(inputs,num_channels,num_channels,ksize = (5,1))


    x1 = tf.nn.avg_pool(inputs,
                       ksize=[1, 4, 1, 1],
                       strides=[1, 4, 1, 1],
                       padding='SAME')
    cx1 = tf.reduce_mean(tf.square(x1))
    x1 = tf.div(x1, cx1)

    # x1 = tf.image.resize_images(x1, size = [18, 16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    x2 = tf.nn.avg_pool(x2,
                        ksize=[1, 4, 1, 1],
                        strides=[1, 4, 1, 1],
                        padding='SAME')
    cx2 = tf.reduce_mean(tf.square(x2))
    x2 = tf.div(x2, cx2)

    #x2 = tf.image.resize_images(x2, size=[18, 16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    x3 = tf.nn.avg_pool(x3,
                        ksize=[1, 2, 1, 1],
                        strides=[1, 2, 1, 1],
                        padding='SAME')
    cx3 = tf.reduce_mean(tf.square(x3))
    x3 = tf.div(x3, cx3)

    #x3 = tf.image.resize_images(x3, size=[18, 16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


    #x1 = tf.nn.relu(x1)

    x = tf.concat([x,x1,x2,x3],3)
    x = conv(x, x.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1))
    logits = tf.reduce_mean(x,axis=2)
    # x_shape = x.get_shape().as_list()
    # outputs = tf.reshape(x, [-1,x_shape[2]*x_shape[3]])
    # W1 = tf.Variable(tf.truncated_normal([x_shape[2]*x_shape[3],
    #                                      150],
    #                                     stddev=0.1))
    # b1 = tf.Variable(tf.constant(0., shape=[150]))
    # # [batch_size*max_timesteps,num_classes]
    # x = tf.matmul(outputs, W1) + b1
    # x= tf.layers.dropout(x)
    # x = tf.nn.relu(x)
    # W2 = tf.Variable(tf.truncated_normal([150,
    #                                      NUM_CHARS+1],
    #                                     stddev=0.1))
    # b2 = tf.Variable(tf.constant(0., shape=[NUM_CHARS+1]))
    # x = tf.matmul(x, W2) + b2
    # x = tf.layers.dropout(x)
    # # [batch_size,max_timesteps,num_classes]
    # logits = tf.reshape(x, [b, -1, NUM_CHARS+1])

    return logits, inputs, targets, seq_len

def train(a):

    train_gen = TextImageGenerator(img_dir=ti,
                                   label_file=tl,
                                   batch_size=BATCH_SIZE,
                                   img_size=img_size,
                                   num_channels=num_channels,
                                   label_len=label_len)

    val_gen = TextImageGenerator(img_dir=vi,
                                 label_file=vl,
                                 batch_size=BATCH_SIZE,
                                 img_size=img_size,
                                 num_channels=num_channels,
                                 label_len=label_len)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               DECAY_STEPS,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    logits, inputs, targets, seq_len = get_train_model(num_channels, label_len,BATCH_SIZE, img_size)
    logits = tf.transpose(logits, (1, 0, 2))
    # tragets是一个稀疏矩阵
    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)

    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=MOMENTUM).minimize(cost, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    # 前面说的划分块之后找每块的类属概率分布，ctc_beam_search_decoder方法,是每次找最大的K个概率分布
    # 还有一种贪心策略是只找概率最大那个，也就是K=1的情况ctc_ greedy_decoder
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    init = tf.global_variables_initializer()

    def report_accuracy(decoded_list, test_targets):
        original_list = decode_sparse_tensor(test_targets)
        detected_list = decode_sparse_tensor(decoded_list)
        true_numer = 0

        if len(original_list) != len(detected_list):
            print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
                  " test and detect length desn't match")
            return
        print("T/F: original(length) <-------> detectcted(length)")
        for idx, number in enumerate(original_list):
            detect_number = detected_list[idx]
            hit = (number == detect_number)
            print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
            if hit:
                true_numer = true_numer + 1
        print("Test Accuracy:", true_numer * 1.0 / len(original_list))

    def do_report(val_gen,num):
        for i in range(num):
            test_inputs, test_targets, test_seq_len = val_gen.next_batch()
            test_feed = {inputs: test_inputs,
                        targets: test_targets,
                        seq_len: test_seq_len}
            st =time.time()
            dd= session.run(decoded[0], test_feed)
            tim = time.time() -st
            print('time:%s'%tim)
            report_accuracy(dd, test_targets)

    def test_report(testi,files):
        true_numer = 0
        num = files//BATCH_SIZE

        for i in range(num):
            test_inputs, test_targets, test_seq_len = val_gen.next_batch()
            test_feed = {inputs: test_inputs,
                        targets: test_targets,
                        seq_len: test_seq_len}
            dd = session.run([decoded[0]], test_feed)
            original_list = decode_sparse_tensor(test_targets)
            detected_list = decode_sparse_tensor(dd)
            for idx, number in enumerate(original_list):
                detect_number = detected_list[idx]
                hit = (number == detect_number)
                if hit:
                    true_numer = true_numer + 1
        print("Test Accuracy:", true_numer * 1.0 / files)


    def do_batch(train_gen,val_gen):
        train_inputs, train_targets, train_seq_len = train_gen.next_batch()

        feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}

        b_loss, b_targets, b_logits, b_seq_len, b_cost, steps, _ = session.run(
            [loss, targets, logits, seq_len, cost, global_step, optimizer], feed)

        #print(b_cost, steps)
        if steps > 0 and steps % REPORT_STEPS == 0:
            do_report(val_gen,test_num)
            saver.save(session, "./model/LPRtf3.ckpt", global_step=steps)
        return b_cost, steps

    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        if a=='train':
            for curr_epoch in range(num_epochs):
                print("Epoch.......", curr_epoch)
                train_cost = train_ler = 0
                for batch in range(BATCHES):
                    start = time.time()
                    c, steps = do_batch(train_gen,val_gen)
                    train_cost += c * BATCH_SIZE
                    seconds = time.time() - start
                    #print("Step:", steps, ", batch seconds:", seconds)

                train_cost /= TRAIN_SIZE
                val_cs=0
                val_ls =0
                for i in range(test_num):
                    train_inputs, train_targets, train_seq_len = val_gen.next_batch()
                    val_feed = {inputs: train_inputs,
                                targets: train_targets,
                                seq_len: train_seq_len}

                    val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)
                    val_cs+=val_cost
                    val_ls+=val_ler

                log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
                print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, train_ler, val_cs/test_num, val_ls/test_num,
                                 time.time() - start, lr))
        if a =='test':
            testi='valid'
            saver.restore(session, './model8.24best/LPRtf3.ckpt-25000')
            test_gen = TextImageGenerator(img_dir=testi,
                                           label_file=None,
                                           batch_size=BATCH_SIZE,
                                           img_size=img_size,
                                           num_channels=num_channels,
                                           label_len=label_len)
            do_report(test_gen,3)


if __name__ == "__main__":
        a = input('train or test:')
        train(a)
