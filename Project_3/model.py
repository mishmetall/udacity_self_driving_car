import tensorflow as tf

class Iter():
    def __init__(self):
        self.iter = 0
    @property
    def get(self):
        self.iter = self.iter+1
        return self.iter - 1


class LeNet():
    def __init__(self, in_shape=[None, 32, 32, 3]):
        self.init_shape = in_shape
        self.count = Iter()

    @staticmethod
    def _estimate_out_shape(n, f, p, s):
        '''
        Calculate output size of image after conv layer

        :param n:image size
        :param f:filter size
        :param p:padding
        :param s:strides
        :return:output size
        '''
        return int((n + 2.0 * p - f)/s) +1

    def build_model(self):
        input = tf.placeholder(shape=self.init_shape, dtype=tf.float32)

        # 32x32x3
        conv_1 = self.convolution2D(1, input, 6, 5, 1, 'VALID')
        print("Conv_1_Layer: ", conv_1.get_shape())
        # 28x28x6

        maxp_1c= tf.nn.max_pool(conv_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        print("MaxPull_1: ", maxp_1c.get_shape())


        # 14x14x6

        conv_2 = self.convolution2D(2, maxp_1c, 16, 5, 1, padding='VALID')

        print("Conv_2_Layer: ", conv_2.get_shape())

        maxp_2 = tf.nn.max_pool(conv_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

        print("max_2_Layer: ", maxp_2.get_shape())

        fl = tf.layers.flatten(maxp_2)
        print("After flatten: ", fl.get_shape())

        fc_1 = self.fully_conected(3, fl, 120)

        fc_2 = self.fully_conected(4, fc_1, 84)

        # TODO: Do i need relu here
        fc_3 = self.fully_conected(4, fc_2, 10)

        return fc_2

    def convolution2D(self, numOfLayer, inputIm, filters, size, stride, padding='SAME', is_batch_norm=True):
        """

        :param numOfLayer: number of convolutional layer
        :param inputIm: input tensor  [batches, height, width, channels]
        :param filters: number of filters
        :param size:    filter size
        :param stride: stride
        :return: convolution with bias + batch normalization + relu

        """
        channels = inputIm.get_shape()[3]

        with tf.variable_scope(str(numOfLayer) + "_convlayer"):
            with tf.variable_scope(str(numOfLayer) + "_conv"):
                filtervalues = tf.Variable(tf.truncated_normal([size, size, int(channels), filters], stddev=0.001))
                biasis = tf.Variable(tf.constant(0.0, shape=[filters]))
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

    def fully_conected(self, numOfLayer, input, out_size):
        # TODO: add scopes
        # TODO: Add batch norm
        # assert len(input.get_shape()) == 1
        print("Len: ", input.get_shape()[1])
        insize = input.get_shape()[1]
        fc_w = tf.Variable(tf.truncated_normal([int(insize), out_size], stddev=0.001))
        fc_b = tf.Variable(tf.zeros(out_size))
        ret = tf.matmul(input, fc_w) + fc_b
        return tf.nn.relu(ret, name=str(numOfLayer)+"_relu")

    def batch_normal(self,numOfLayer, input):
        '''
        I want to compare it with tf.layers.batch_normalization
        :param numOfLayer:
        :param input:
        :return:
        '''
        size = input.shape[0]
        mean, var = tf.nn.moments(input, [0])
        beta = tf.Variable(tf.zeros(size))
        scale = tf.Variable(tf.zeros(size))
        with tf.variable_scope(str(numOfLayer) + "_batch"):
            ret = tf.nn.batch_normalization(input, mean, var, beta, scale, 0.001)
        return ret

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

    def get_graph_sammary(self, sess_graph):

        self.summary_writer = tf.summary.FileWriter("lenet", sess_graph)

        # self.summary_los = tf.summary.scalar("loss by step", self.loss)

        # self.summary_im = tf.summary.image("images",self.classify)

        self.summary = tf.summary.merge_all()



if __name__ == '__main__':
    mod = LeNet().build_model()
