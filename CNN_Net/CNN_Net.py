import tensorflow as tf
import Transform_Data_v1 as td
import numpy as np
import time

sess=tf.InteractiveSession()

Epoch=300
Test_epoch=10
L2_parameter=1e-4
constant_initializer_parameter=0.1
with tf.name_scope('Learning_Rate'):
    learning_rate=tf.Variable(1e-4,name='learning_rate')
    tf.summary.scalar(' ',learning_rate)

max_val_accuracy=0.0
counter=0
stop_counter=0

training_batch_size=128
validation_batch_size=128
test_batch_size=128

Load_Data=False
Training=True

model_name='Cnn_Net'
checkpoint_dir='D:/Checkpoint/' + model_name + '/'

len_training_data, validation, test = td.Initialization(validation_batch_size,test_batch_size)
iteration_time=int(len_training_data/training_batch_size)

def weight_variable(shape, Name):
    with tf.variable_scope(Name):
        initial = tf.get_variable('weight',shape,initializer=tf.contrib.layers.xavier_initializer())
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(L2_parameter)(initial))
        return initial

def bias_variable(shape,Name):
    with tf.variable_scope(Name):
        initial=tf.get_variable('bias',shape,initializer=tf.constant_initializer(constant_initializer_parameter))
        return initial

def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def layer_build(input, conv_size, in_channel, out_channel, Layer_num, pooling=False):
    with tf.variable_scope('ConvLayer_' + str(Layer_num)):
            weight = weight_variable([conv_size, conv_size, in_channel, out_channel], 'Weight')
            biases = bias_variable([out_channel], 'Bias')
            conv = tf.nn.bias_add(conv2d(input, weight), biases)
            conv_result_bn=tf.layers.batch_normalization(conv)
            tf.summary.histogram(' ',conv_result_bn)
            conv_result = tf.nn.relu(conv_result_bn)
            if pooling == True:
                    conv_result = max_pool_2x2(conv_result)
            return conv_result

def test_procedure():
    confusion_matrics=np.zeros([7,7],dtype="int")

    for k in range(len(test)):
        matrix_row, matrix_col = sess.run(distribution, feed_dict={x: test[k][0], y_: test[k][1]})
        for m, n in zip(matrix_row, matrix_col):
            confusion_matrics[m][n] += 1

    test_accuracy=float(np.sum([confusion_matrics[q][q] for q in range(7)])/np.sum(confusion_matrics))

    log1 = "Test Accuracy : %g" % test_accuracy
    log2 = np.array(confusion_matrics.tolist())

    logfile = open(checkpoint_dir + model_name + '.txt', 'a+')
    print(log1)
    print(log2)
    print(log1,file=logfile)
    print(log2,file=logfile)
    logfile.close()

with tf.name_scope('Input'):
    with tf.name_scope('Input_x'):
        x = tf.placeholder(tf.float32,shape=[None,1024])
    with tf.name_scope('Input_y'):
        y_ = tf.placeholder(tf.float32,shape=[None,7])
    with tf.name_scope('Reshape_x'):
        x_image = tf.reshape(x, [-1, 32, 32, 1])
        tf.summary.image('input_reshape',x_image,max_outputs=3)


conv1=layer_build(input=x_image, conv_size=3, in_channel=1,   out_channel=64,  Layer_num=1,   pooling=False)
conv2=layer_build(input=conv1,   conv_size=3, in_channel=64,  out_channel=64,  Layer_num=2,   pooling=True)

conv3=layer_build(input=conv2,   conv_size=3, in_channel=64,  out_channel=128, Layer_num=3,   pooling=False)
conv4=layer_build(input=conv3,   conv_size=3, in_channel=128, out_channel=128, Layer_num=4,   pooling=True)

conv5=layer_build(input=conv4,   conv_size=3, in_channel=128, out_channel=256, Layer_num=5,   pooling=False)
conv6=layer_build(input=conv5,   conv_size=3, in_channel=256, out_channel=256, Layer_num=6,   pooling=False)
conv7=layer_build(input=conv6,   conv_size=3, in_channel=256, out_channel=256, Layer_num=7,   pooling=False)
conv8=layer_build(input=conv7,   conv_size=3, in_channel=256, out_channel=256, Layer_num=8,   pooling=True)

conv9=layer_build(input=conv8,   conv_size=3, in_channel=256, out_channel=512, Layer_num=9,   pooling=False)
conv10=layer_build(input=conv9,  conv_size=3, in_channel=512, out_channel=512, Layer_num=10,  pooling=False)
conv11=layer_build(input=conv10, conv_size=3, in_channel=512, out_channel=512, Layer_num=11,  pooling=False)
conv12=layer_build(input=conv11, conv_size=3, in_channel=512, out_channel=512, Layer_num=12,  pooling=True)

conv13=layer_build(input=conv12, conv_size=3, in_channel=512, out_channel=512, Layer_num=13,  pooling=False)
conv14=layer_build(input=conv13, conv_size=3, in_channel=512, out_channel=512, Layer_num=14,  pooling=True)
# conv15=layer_build(input=conv14, conv_size=3, in_channel=512, out_channel=512, Layer_num=15,  pooling=False)
# conv16=layer_build(input=conv15, conv_size=3, in_channel=512, out_channel=512, Layer_num=16,  pooling=True)


with tf.variable_scope('Full_Connected_Layer1'):
    w_fc1=weight_variable([1*1*512,1024],'weight')
    b_fc1=bias_variable([1024],'bias')
    h_pool_flat=tf.reshape(conv14,[-1,1*1*512])
    h_fc1_activation=tf.nn.bias_add(tf.matmul(h_pool_flat,w_fc1), b_fc1)
    h_fc1_bn = tf.layers.batch_normalization(h_fc1_activation)
    h_fc1=tf.nn.relu(h_fc1_bn)

with tf.variable_scope('Full_Connected_Layer2'):
    w_fc2=weight_variable([1024,7],'weight')
    b_fc2=bias_variable([7],'bias')
    y_conv=tf.nn.bias_add(tf.matmul(h_fc1,w_fc2), b_fc2)
    y_conv=tf.layers.batch_normalization(y_conv)

with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    tf.add_to_collection('losses', cross_entropy)
    loss = tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('Loss_Function', loss)

with tf.name_scope('Train_step'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.name_scope('Accuracy'):
    distribution=[tf.arg_max(y_,1),tf.arg_max(y_conv,1)]
    correct_prediction=tf.equal(distribution[0],distribution[1])

    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
    tf.summary.scalar('Accuracy',accuracy)

saver=tf.train.Saver()

merged=tf.summary.merge_all()
writer=tf.summary.FileWriter(checkpoint_dir,sess.graph)

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

        train_accuracy, losses, result = sess.run([accuracy, loss, merged], feed_dict={x: batch[0], y_: batch[1]})

        validation_accuracy_matrix=np.zeros([7,7],dtype="int")

        for j in range(len(validation)):
            val_acc_row, val_acc_col = sess.run(distribution,
                                                feed_dict={x: validation[j][0], y_: validation[j][1]})
            for r,c in zip(val_acc_row,val_acc_col):
                validation_accuracy_matrix[r][c] += 1
        validation_accuracy=float(np.sum([validation_accuracy_matrix[p][p] for p in range(7)])/np.sum(validation_accuracy_matrix))

        log = "Epoch %d , training accuracy %g,Validation Accuracy: %g, Losses : %g , learning rate: % g time : %s" % \
              (e, train_accuracy, validation_accuracy, losses, sess.run(learning_rate),time.ctime(time.time()))

        logfile = open(checkpoint_dir + model_name + '.txt', 'a+')
        print(log)
        print(log, file=logfile)
        logfile.close()

        writer.add_summary(result, e)
        saver.save(sess,checkpoint_dir+ model_name +'.ckpt',global_step=e)

        if validation_accuracy > max_val_accuracy:
            max_val_accuracy = validation_accuracy
            counter = 0
        else:
            counter += 1
            if counter == 20:
                learning_rate /= 10
                print('learning rate changed !')
                counter = 0
                stop_counter += 1
                print('stop counter : %d' % stop_counter)
            if stop_counter == 3:
                print('training stop')
                break

        if e % Test_epoch == 0:
            test_procedure()

print('final test :')
test_procedure()
sess.close()