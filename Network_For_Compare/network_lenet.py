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
constant_initializer_parameter=0.1

max_val_accuracy=0.0
counter=0

training_batch_size = 128
test_batch_size = 128

Load_Data=False
Training=True
Early_stop=False

model_name='Network_Lenet_2'
checkpoint_dir='D:/Checkpoint/' + model_name + '/'

len_training_data, validation, test = td.Initialization(test_batch_size)
iteration_time=int(len_training_data/training_batch_size)+1

with tf.name_scope('Learning_Rate'):
    learning_rate=tf.constant(start_learning_rate)
    tf.summary.scalar(' ',learning_rate)

def conv_layer(input, conv_ksize, in_channel, out_channel, layer_name):
    with tf.variable_scope(layer_name):
        weight = tf.get_variable('weight',shape=[conv_ksize, conv_ksize, in_channel, out_channel],initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', shape=[out_channel], initializer=tf.constant_initializer(constant_initializer_parameter))
        conv = tf.nn.bias_add(tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding='VALID'), bias)
        conv = tf.nn.relu(conv)
        tf.summary.histogram(' ',conv)
    return conv

def max_pool(x, size):
    with tf.variable_scope('Max_Pooling'):
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

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
        matrix_row, matrix_col = sess.run(distribution, feed_dict={x: test[k][0], y_: test[k][1]})
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

conv = conv_layer(input=x_reshape, conv_ksize=5, in_channel=1, out_channel=8, layer_name='conv1')
conv = max_pool(x=conv, size=2)
conv = conv_layer(input=conv, conv_ksize=5, in_channel=8, out_channel=20, layer_name='conv2')
conv = max_pool(x=conv, size=2)
conv = conv_layer(input=conv, conv_ksize=5, in_channel=20, out_channel=120, layer_name='conv3')


with tf.variable_scope('Reshape'):
    conv_reshape = tf.reshape(conv,[-1,1*1*120])

with tf.variable_scope('FC'):
    weight = tf.get_variable('weight', shape=[120,7],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
    bias = tf.get_variable('bias', shape=[7],
                           initializer=tf.constant_initializer(constant_initializer_parameter))
    y_conv = tf.nn.bias_add(tf.matmul(conv_reshape,weight),bias)
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
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
            # print('Step %d / %d in Epoch %d finished !' % (iteration, iteration_time, e))

        training_accuracy, training_loss, result_training = sess.run([accuracy, loss, merged], feed_dict={x: batch[0], y_: batch[1]})
        validation_accuracy, validation_loss, result_validation = sess.run([accuracy, loss, merged],feed_dict={x: validation[0], y_: validation[1]})

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