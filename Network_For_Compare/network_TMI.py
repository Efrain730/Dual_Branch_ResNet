import tensorflow as tf
import Transform_Data_v1 as td
import numpy as np
import time

# Weight Initializer is truncated_normal initializer

# config=tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess=tf.InteractiveSession(config=config)

sess = tf.InteractiveSession()

Epoch = 200
Test_epoch=10
start_learning_rate = 1e-3
constant_initializer_parameter=0.0
decay_rate=0.98
k=4
dropout = 0.5
leaky_alpha = 0.3

max_val_accuracy=0.0
counter=0

training_batch_size = 128
test_batch_size = 128

Load_Data=False
Training=True
Early_stop=False

model_name='Network_TMI'
checkpoint_dir='D:/Checkpoint/' + model_name + '/'

len_training_data, validation, test = td.Initialization(test_batch_size)
iteration_time=int(len_training_data/training_batch_size)+1

with tf.name_scope('Learning_Rate'):
    learning_rate=tf.constant(start_learning_rate)
    tf.summary.scalar(' ',learning_rate)

def conv_layer(input, conv_ksize, in_channel, out_channel, layer_name):
    with tf.variable_scope(layer_name):
        weight = tf.get_variable('weight',shape=[conv_ksize, conv_ksize, in_channel, out_channel],initializer=tf.orthogonal_initializer(gain=1.1))
        bias = tf.get_variable('bias', shape=[out_channel], initializer=tf.constant_initializer(constant_initializer_parameter))
        conv = tf.nn.bias_add(tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding='VALID'), bias)
        conv = leaky_relu(conv)
        tf.summary.histogram(' ',conv)
    return conv

def leaky_relu(x, alpha=leaky_alpha):
    with tf.variable_scope('Leaky_Relu'):
        return tf.nn.relu(x)-alpha*tf.nn.relu(-x)

def avg_pool(x, size):
    with tf.variable_scope('Average_Pooling'):
        return tf.nn.avg_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

def f_value(matrix):
    f = 0.0
    length = len(matrix[0])
    for i in range(length):
        recall = matrix[i][i] / np.sum([matrix[i][m] for m in range(7)])
        precision = matrix[i][i] / np.sum([matrix[n][i] for n in range(7)])
        result = (recall*precision) / (recall+precision)
        print(result)
        f += result
    f *= (2/7)
    return f

def test_procedure():
    confusion_matrics=np.zeros([7,7],dtype="int")

    for k in range(len(test)):
        matrix_row, matrix_col = sess.run(distribution, feed_dict={x: test[k][0], y_: test[k][1], keep_prob:1.0})
        for m, n in zip(matrix_row, matrix_col):
            confusion_matrics[m][n] += 1

    test_accuracy=float(np.sum([confusion_matrics[q][q] for q in range(7)])/np.sum(confusion_matrics))
    detail_test_accuracy = [confusion_matrics[i][i]/np.sum(confusion_matrics[i]) for i in range(7)]
    log1 = "Test Accuracy : %g" % test_accuracy
    log2 = np.array(confusion_matrics.tolist())
    log3= ''
    for j in range(7):
        log3 += 'category %g test accuracy : %g\n' % (j,detail_test_accuracy[j])
    logfile = open(checkpoint_dir + model_name+ '.txt', 'a+')
    log4 = 'F_Value : %g' % f_value(confusion_matrics)
    print(log1)
    print(log2)
    print(log3)
    print(log4)
    print(log1,file=logfile)
    print(log2,file=logfile)
    print(log3, file=logfile)
    print(log4, file=logfile)
    logfile.close()

with tf.name_scope('Input'):
    with tf.name_scope('Input_x'):
        x = tf.placeholder(tf.float32,shape=[None,1024])
    with tf.name_scope('Input_y'):
        y_ = tf.placeholder(tf.float32,shape=[None,7])
    with tf.name_scope('Image_Reshape'):
        x_reshape = tf.reshape(x, [-1,32,32,1])

with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(tf.float32)

conv = conv_layer(input=x_reshape, conv_ksize=2, in_channel=1, out_channel=4*k, layer_name='conv1')
conv = conv_layer(input=conv, conv_ksize=2, in_channel=4*k, out_channel=9*k, layer_name='conv2')
conv = conv_layer(input=conv, conv_ksize=2, in_channel=9*k, out_channel=16*k, layer_name='conv3')
conv = conv_layer(input=conv, conv_ksize=2, in_channel=16*k, out_channel=25*k, layer_name='conv4')
conv = conv_layer(input=conv, conv_ksize=2, in_channel=25*k, out_channel=36*k, layer_name='conv5')

with tf.variable_scope('Trans_Dropout'):
    avg_conv = avg_pool(conv, conv.get_shape()[1].value)
    avg_conv_reshape = tf.reshape(avg_conv,[-1,1*1*36*k])
    avg_conv_reshape_dropout = tf.nn.dropout(avg_conv_reshape, keep_prob=keep_prob)

with tf.variable_scope('FC1'):
    weight = tf.get_variable('weight', shape=[avg_conv_reshape_dropout.get_shape()[1].value,6*36*k],
                             initializer=tf.uniform_unit_scaling_initializer(factor=np.sqrt(32*32)))
    bias = tf.get_variable('bias', shape=[6*36*k],
                           initializer=tf.constant_initializer(constant_initializer_parameter))
    fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(avg_conv_reshape_dropout, weight),bias))
    tf.summary.histogram(' ', fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

with tf.variable_scope('FC2'):
    weight = tf.get_variable('weight', shape=[6*36*k,2*36*k],
                             initializer=tf.uniform_unit_scaling_initializer(factor=np.sqrt(32*32)))
    bias = tf.get_variable('bias', shape=[2*36*k],
                           initializer=tf.constant_initializer(constant_initializer_parameter))
    fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc1, weight), bias))
    tf.summary.histogram(' ', fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)

with tf.variable_scope('FC3'):
    weight = tf.get_variable('weight', shape=[2*36*k,7],
                             initializer=tf.uniform_unit_scaling_initializer(factor=np.sqrt(32*32)))
    bias = tf.get_variable('bias', shape=[7],
                           initializer=tf.constant_initializer(constant_initializer_parameter))
    y_conv = tf.nn.bias_add(tf.matmul(fc2,weight),bias)
    tf.summary.histogram(' ',y_conv)

with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    tf.add_to_collection('losses', cross_entropy)

    loss = tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('Loss', loss)

with tf.name_scope('Train_step'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.name_scope('Accuracy'):
    distribution=[tf.arg_max(y_,1),tf.arg_max(y_conv,1)]
    correct_prediction=tf.equal(distribution[0],distribution[1])

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    tf.summary.scalar('Accuracy',accuracy)

saver=tf.train.Saver(max_to_keep=30)

merged=tf.summary.merge_all()
writer_training=tf.summary.FileWriter(checkpoint_dir+'train/',sess.graph)
writer_validation=tf.summary.FileWriter(checkpoint_dir+'validation/',sess.graph)

if Load_Data==True:
    ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

if Training:
    for e in range(1,1+Epoch):
        for iteration in range(iteration_time):
            batch = td.next_batch(training_batch_size)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob:dropout})
            # print('Step %d / %d in Epoch %d finished !' % (iteration, iteration_time, e))

        training_accuracy, training_loss, result_training = sess.run([accuracy, loss, merged], feed_dict={x: batch[0], y_: batch[1], keep_prob:dropout})
        validation_accuracy, validation_loss, result_validation = sess.run([accuracy, loss, merged],feed_dict={x: validation[0], y_: validation[1], keep_prob:1.0})

        log = "Epoch %d , training accuracy %g ,Validation Accuracy: %g , Loss_training : %g , Loss_validation: %g , learning rate: % g time: %s" % \
              (e, training_accuracy, validation_accuracy, training_loss, validation_loss, sess.run(learning_rate),time.ctime(time.time()))

        logfile = open(checkpoint_dir + model_name + '.txt', 'a+')
        print(log)
        print(log, file=logfile)
        logfile.close()

        writer_training.add_summary(result_training, e)
        writer_validation.add_summary(result_validation,e)
        saver.save(sess,checkpoint_dir+ model_name + '.ckpt',global_step=e)

        if Early_stop:
            if validation_accuracy > max_val_accuracy:
                max_val_accuracy = validation_accuracy
                counter=0
            else:
                counter+=1
                if counter == 30:
                    break

        if e % Test_epoch == 0:
            test_procedure()

print('final test :')
test_procedure()
sess.close()