import tensorflow as tf

class Iter():
    def __init__(self):
        self.iter = 0

    @property
    def get(self):
        self.iter = self.iter+1
        return self.iter - 1


class LeNet():
    def __init__(self, in_shape=[None, 32, 32, 3], classes=43):

        self.init_shape = in_shape
        self.number_classes = int(classes)

        ## PlaceHolders

        self.input = tf.placeholder(shape=self.init_shape, dtype=tf.float32)
        self.true = tf.placeholder(shape=[None, self.number_classes], dtype=tf.float32)
        self.ln_rate = tf.placeholder(dtype=tf.float32)

        ###############################################################################
        # self.true = tf.placeholder(shape=(None), dtype=tf.int32)
        # self.true = tf.one_hot(self.true, self.number_classes, dtype=tf.float32)

        self.model = self.__build_model()

        self.loss = self._loss()

        self.optimizer = self._int_optimize()

        #### FOR DEBUG
        # self.cross=None

    def __build_model(self):


        with tf.variable_scope('Block_1'):
            conv_1 = self.convolution2D(1, self.input, 6, 5, 1, 'VALID')
            print("Conv Block 1: ", conv_1.get_shape(), " Weights: ", self.trainable_variable())
            maxp_1c= tf.nn.max_pool(conv_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
            print("MaxPulling 1: ", maxp_1c.get_shape(), " Weights: ", self.trainable_variable())


        # 14x14x6
        with tf.variable_scope('Block_2'):
            conv_2 = self.convolution2D(2, maxp_1c, 16, 5, 1, padding='VALID')
            print("Conv Block 2: ", conv_2.get_shape(), " Weights: ", self.trainable_variable())
            maxp_2 = tf.nn.max_pool(conv_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
            print("MaxPulling 2: ", maxp_2.get_shape(), " Weights: ", self.trainable_variable())

        with tf.variable_scope('Flatten'):
            fl = tf.layers.flatten(maxp_2)
            print("Flatten: ", fl.get_shape(), " Weights: ", self.trainable_variable())

        with tf.variable_scope('Block_3'):
            fc_1 = self.fully_conected(3, fl, 320)
            print("Fully Block 1: ", fc_1.get_shape(), " Weights: ", self.trainable_variable())

        with tf.variable_scope('Block_4'):
            fc_2 = self.fully_conected(4, fc_1, 240)
            print("Fully Block 2: ", fc_2.get_shape(), " Weights: ", self.trainable_variable())

        # TODO: Do i need relu here
        with tf.variable_scope('Block_5'):
            fc_3 = self.fully_conected(5, fc_2, self.number_classes, activation=False)
            print("Fully Block 3: ", fc_3.get_shape(), " Weights: ", self.trainable_variable())

        return fc_3

    def _loss(self):
        self.cross = tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.true)
        loss = tf.reduce_mean(self.cross)
        return loss


    def _int_optimize(self):
        optimizer = tf.train.AdagradOptimizer(learning_rate=self.ln_rate)
        return optimizer.minimize(self.loss)

    def convolution2D(self, numOfLayer, inputIm, filters, size, stride, padding='SAME', is_batch_norm=False):
        """
        Instead i should use tf.keras.layers
        but its just for clarification
        :param numOfLayer: number of convolutional layer
        :param inputIm: input tensor  [batches, height, width, channels]
        :param filters: number of filters
        :param size:    filter size
        :param stride: stride
        :return: convolution with bias + batch normalization + relu

        """
        channels = inputIm.get_shape()[3]

        with tf.variable_scope(str(numOfLayer) + "_convlayer"):
            filtervalues = tf.Variable(tf.truncated_normal([size, size, int(channels), filters], stddev=0.1))
            biasis = tf.Variable(tf.constant(0.0, shape=[filters]), trainable=True)
            conv = tf.nn.conv2d(inputIm, filtervalues, strides=[1, stride, stride, 1], padding=padding,
                                name=str(numOfLayer) + "_conv")
            conv = tf.add(conv, biasis)



            # Relu
            # TODO: figure out what first activation or normalization
            relu = tf.nn.relu(conv, name=str(numOfLayer) + "_relu")

            if is_batch_norm == True:
                with tf.variable_scope(str(numOfLayer) + "_batch"):
                    relu = tf.layers.batch_normalization(inputs=conv, axis=-1)
            return relu

    def fully_conected(self, numOfLayer, input, out_size, activation=True):
        # TODO: add scopes
        # TODO: Add batch norm
        # assert len(input.get_shape()) == 1
        with tf.variable_scope(str(numOfLayer) + "_fully"):
            insize = input.get_shape()[1]
            fc_w = tf.Variable(tf.truncated_normal([int(insize), out_size], stddev=0.1))
            fc_b = tf.Variable(tf.zeros(out_size))
            ret = tf.matmul(input, fc_w) + fc_b
            return ret if activation == False else tf.nn.relu(ret)

    # def batch_normal(self,numOfLayer, input):
    #     '''        # I did this function just to understand am I right about my batch norm intuition
    #
    #
    #     I want to compare it with tf.layers.batch_normalization
    #     :param numOfLayer:
    #     :param input:
    #     :return:
    #     '''
    #     size = input.shape[0]
    #     mean, var = tf.nn.moments(input, [0])
    #     beta = tf.Variable(tf.zeros(size))
    #     scale = tf.Variable(tf.ones(size))
    #     with tf.variable_scope(str(numOfLayer) + "_batch"):
    #         ret = tf.nn.batch_normalization(input, mean, var, beta, scale, 0.0001)
    #     return ret

    def trainable_variable(self, scope=None):

        # Input:
        #   variable scope
        # Return:
        #   number of trainable parametrs
        #   in corresponding variable scope
        #   defaul number of parametrs from all graph

        total_parameters = 0
        weight = tf.trainable_variables(scope=scope)
        for variable in weight:
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

    def set_graph_sammary(self, sess_graph):

        self.summary_writer = tf.summary.FileWriter("lenet", sess_graph)

        self.summary_los = tf.summary.scalar("loss by step", self.loss)

        # self.summary_im = tf.summary.image("images",self.classify)

        self.summary = tf.summary.merge_all()

#
# def training():
#
#     mod = LeNet().build_model()

if __name__ == '__main__':
    mod = LeNet()

