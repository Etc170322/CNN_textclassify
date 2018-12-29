import tensorflow as tf
import os
import time,random
from gensim.models import KeyedVectors
import jieba

default_dir_path = os.path.split(os.path.abspath(__file__))[0]

def read_embedding(model_path):
    model = KeyedVectors.load_word2vec_format(model_path)
    return model

def train_step(x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: 0.5
    }
    _, step, summaries, loss, accuracy = sess.run(
        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = time.time()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    train_summary_writer.add_summary(summaries, step)

def dev_step(x_batch, y_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: 1.0
    }

    step, summaries, loss, accuracy = sess.run(
        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = time.time()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    if writer:
        writer.add_summary(summaries, step)

def read_data(data_path):
    class_dir = os.listdir(data_path)
    corpus=[]
    for class_i in class_dir:
        abs_dir = os.path.join(data_path,class_i)
        list_files = os.listdir(abs_dir)
        for file in list_files:
            abs_file_path = os.path.join(abs_dir,file)
            with open(abs_file_path,'r',encoding='utf-8') as f:
                lines = f.readlines()
                words=[]
                for line in lines:
                    words.extend(list(jieba.cut(line.strip())))
                if words == []:
                    continue
                label = [0]*len(class_dir)
                label[int(class_i)]=1
                corpus.append([embedding_lookup(embedding_model, words, 128),label])
    return corpus

def batch_data(corpus,batch_size=None,num_epochs=1):
    # random.shuffle(corpus)
    big_corpus = corpus*num_epochs
    random.shuffle(big_corpus)
    sum_corpus = len(big_corpus)
    x_data = [item[0] for item in big_corpus]
    y_data = [item[1] for item in big_corpus]
    batches=[]
    if batch_size == None:
        return [x_data,y_data]
    else:
        num_batches = int(sum_corpus/batch_size)
        for i in range(num_batches-1):
            batches.append([x_data[i*batch_size:(i+1)*batch_size],y_data[i*batch_size:(i+1)*batch_size]])
    return batches

def embedding_lookup(word_embedding,words,max_length):
        vectors=[]
        i = 0
        for word in words:
            if i == max_length:
                break
            try:
                vectors.append(word_embedding.wv[word])
                i+=1
            except:
                continue
        if i < max_length:
            for j in range(i,max_length):
                vectors.append([0]*len(vectors[0]))
        return vectors

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters,word_embedding):
        """
        :param sequence_length: 我们的句子的长度。 请记住，我们填充所有句子的长度相同（59为我们的数据集）。
        :param num_classes: 输出层中的类数，在我们的例子中为正（负）。
        :param vocab_size: 我们的词汇量。 这需要定义我们的嵌入层的大小，它将具有[vocabulary_size，embedding_size]的形状。
        :param embedding_size: 我们嵌入的维度。
        :param filter_sizes: 我们想要卷积过滤器覆盖的字数。 我们将为此处指定的每个大小设置num_filters。 例如，[3,4,5]意味着我们将有一个过滤器，分别滑过3个，4个和5个字，总共有3 * num_filters过滤器。
        :param num_filters: 每个过滤器大小的过滤器数量（见上文）。
        """
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length,embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.input_x_expanded = tf.expand_dims(self.input_x,3)
        # tf.device（“/ cpu：0”）强制在CPU上执行操作。 默认情况下，TensorFlow将尝试将操作放在GPU上（如果有的话）可用，但是嵌入式实现当前没有GPU支持，并且如果放置在GPU上则会抛出错误。
        # tf.name_scope创建一个名称范围，名称为“embedding”。 范围将所有操作添加到名为“嵌入”的顶级节点中，以便在TensorBoard中可视化您的网络时获得良好的层次结构。
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # W是我们的滤波器矩阵，h是将非线性应用于卷积输出的结果。
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # “VALID”填充意味着我们将过滤器滑过我们的句子而不填充边缘，
                # 执行一个窄的卷积，给出一个形状[1，sequence_length - filter_size + 1,1,1]的输出。
                conv = tf.nn.conv2d(
                    self.input_x_expanded,
                    w,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 在特定过滤器大小的输出上执行最大化池将留下一张张量[batch_size，1，num_filters]。
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 2, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total=0
        for item in pooled_outputs:
            num_filters_total+=item.shape.as_list()[1]
        self.h_pool = tf.concat(pooled_outputs,1)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            w = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

with tf.Graph().as_default():
    sess = tf.Session()
    embedding_model = read_embedding(r'E:\数据\CNN_Numpy\word2vec_model\models\20181228_model_vector.txt')

    with sess.as_default():
        cnn = TextCNN(
            sequence_length=128,
            num_classes=2,
            vocab_size=50000,
            embedding_size=200,
            filter_sizes=[3,4,5],
            num_filters=1,
            word_embedding=embedding_model,
        )
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)


        # Checkpointing
        checkpoint_dir = os.path.abspath(os.path.join(default_dir_path, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        # Tensorflow assumes this directory already exists so we need to create it
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        sess.run(tf.global_variables_initializer())

        data_path = os.path.join(os.path.split(default_dir_path)[0],"NewsData")
        corpus = read_data(os.path.join(data_path,"train"))
        train_batches = batch_data(corpus, batch_size=50, num_epochs=5)

        corpus_dev = read_data(os.path.join(data_path,"dev"))
        x_dev = [item[0] for item in corpus_dev]
        y_dev = [item[1] for item in corpus_dev]

        # Training loop. For each batch...
        for batch in train_batches:
            x_batch, y_batch = batch
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 500 == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % 500 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))



