import os
from dataset import get_data
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


from model import LeNet


BATCH_SIZE = 200
EPOCH = 50
LNRATE = 0.001
PATH2WEIGHTS = "./model.ckpt"


ds_train, ds_valid = get_data()

# init model
Net = LeNet(classes=ds_train.classes)

graph = tf.get_default_graph()
# init sess
sess = tf.Session()
sess.run(tf.global_variables_initializer())



saver = tf.train.Saver()
# if os.path.exists("/home/sviat/Documents/udacity/udacity_self_driving_car/Project_3/model.ckpt.data-00000-of-00001"):
#     saver.restore(sess, "/home/sviat/Documents/udacity/udacity_self_driving_car/Project_3/model.ckpt")
#     print("LOADED WEIGHTS by pass: ", PATH2WEIGHTS)

Net.set_graph_sammary(sess_graph=sess.graph)

epoch_train_losses = []
valid_losses = []
#### MAIN  LOOOOOOOOP
for e in range(EPOCH):

    print("EPOCH: ", e+1, "/", EPOCH)

    ds_train.shuffle()
    losses = []
    for i in range(0,len(ds_train) // BATCH_SIZE):

        x_batch, y_batch = ds_train.get_batch(i * BATCH_SIZE, (i+1)*BATCH_SIZE - 1)
        o, l,c, m = sess.run([Net.optimizer, Net.loss, Net.cross, Net.model],
                             feed_dict={Net.keep_rate: 0.7,
                                        Net.input:x_batch,
                                        Net.true:y_batch,
                                        Net.ln_rate:LNRATE})
        losses.append(l)

    epoch_train_losses.append(np.mean(np.asarray(losses)))
    print("Train loss per epoch: ",epoch_train_losses[-1])

    ################# Validation part  ####################
    losses = []
    tp = 0
    ds_valid.shuffle()
    for i in range(0,len(ds_valid) // BATCH_SIZE):

        x_batch, y_batch = ds_valid.get_batch(i * BATCH_SIZE, (i+1)*BATCH_SIZE - 1)
        l, s, acc = sess.run([Net.loss, Net.summary, Net.accuracy_operation],
                             feed_dict={Net.keep_rate: 1,
                                        Net.input: x_batch,
                                        Net.true:  y_batch})
        Net.summary_writer.add_summary(s, e)
        Net.summary_writer.flush()
        losses.append(l)
        tp += (acc * len(x_batch))

    valid_losses.append(np.mean(np.asarray(losses)))
    print("Valid loss per epoch: ", valid_losses[-1])
    if ( len(valid_losses) > 1 and valid_losses[-1] <= min(valid_losses) ):
        saver.save(sess, PATH2WEIGHTS)
        print("Saved")

    validation_accuracy = tp / len(ds_valid)
    print("EPOCH {} : Validation Accuracy = {:.3f}%".format(i + 1, (validation_accuracy * 100)))


# hist = np.histogram(np.asarray(losses))
#
# import matplotlib.pyplot as plt
#
# plt.plot(epoch_losses, [i for i in range(len(epoch_losses))])
# plt.show()

    # break