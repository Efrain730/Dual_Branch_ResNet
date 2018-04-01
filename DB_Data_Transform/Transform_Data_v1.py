import numpy as np
import FileNameExtraction as FNE

training_percentage = 0.99

training = []

fileNameClasses_training=FNE.extractFiles(FNE.training_path)
fileNameClasses_test=FNE.extractFiles(FNE.test_path)

def split_into_trainingFiles_and_validationFiles(dataSet, trainingFiles_percentage):
    fileForTraining,fileForValidation=[],[]
    for c in dataSet:
        fileForTraining.append(c[:int(len(c)*trainingFiles_percentage)])
        fileForValidation.append(c[int(len(c)*trainingFiles_percentage):])

    return fileForTraining,fileForValidation

def extractInformation(fileName):
    image,label=[],[]
    temp_image=[]
    line_counter=0

    file=open(FNE.training_path+fileName)
    for line in file:
        if line.index('\n')==1:
            label.append(int(line))
            continue
        else:
            str=line[:-2]
            split_str=str.split(',')
            for item in split_str:
                temp_image.append(float(item))
            line_counter+=1
            if line_counter % 32 == 0:
                image.append(temp_image)
                temp_image=[]
    file.close()
    image = np.array(image)
    label_one_hot=np.zeros([len(label),7])
    for i,j in zip(range(len(label)),label):
        label_one_hot[i][j-1]=1

    return image,label_one_hot

def locate_label_position(label):
    for i in range(len(label)):
        if label[i]!=0:
            return i

def collectInformation(fileNameClass,training=False, validation=False):
    if training==True:
        training_image,training_label=[],[]
        for single_class in fileNameClass:
            class_number=0
            class_name = ''
            for single_fileName in single_class:
                temp_image,temp_label=extractInformation(single_fileName)
                class_number += len(temp_image)
                class_name = str(locate_label_position(temp_label[0]))
                for item_temp_image,item_temp_label in zip(temp_image,temp_label):
                    training_image.append(item_temp_image)
                    training_label.append(item_temp_label)
            print('Training Sets class %s have %g images' % (class_name,class_number))

        return training_image,training_label
    elif validation==True:
        validation_image,validation_label=[],[]
        for single_class in fileNameClass:
            class_number=0
            class_name=''
            for single_fileName in single_class:
                temp_image,temp_label=extractInformation(single_fileName)
                class_number += len(temp_image)
                class_name = str(locate_label_position(temp_label[0]))
                for item_temp_image, item_temp_label in zip(temp_image, temp_label):
                    validation_image.append(item_temp_image)
                    validation_label.append(item_temp_label)
            print('Validation Sets class %s have %g images' % (class_name, class_number))

        return validation_image,validation_label
    else:
        test_image, test_label = [], []
        for single_class in fileNameClass:
            class_number = 0
            class_name = ''
            for single_fileName in single_class:
                temp_image, temp_label = extractInformation(single_fileName)
                class_number += len(temp_image)
                class_name = str(locate_label_position(temp_label[0]))
                for item_temp_image, item_temp_label in zip(temp_image, temp_label):
                    test_image.append(item_temp_image)
                    test_label.append(item_temp_label)
            print('Test Sets class %s have %g images' % (class_name, class_number))

        return test_image,test_label

def Initialization(test_batchSize):
    global training
    final_test=[]
    item_final_validation,item_final_test=[],[]

    trainingFileSets, validationFileSets = split_into_trainingFiles_and_validationFiles(fileNameClasses_training, training_percentage)
    testFileSets = fileNameClasses_test
    training=collectInformation(trainingFileSets, training=True)
    validation=collectInformation(validationFileSets,validation=True)
    test=collectInformation(testFileSets)

    print('Processing three kinds of data finished !')
    training_size=len(training[0])
    validation_size=len(validation[0])
    test_size=len(test[0])
    print('Number of training data : %d, validation data : %d, test data %d' % (training_size, validation_size, test_size))
    test_iteration_time = int(test_size / test_batchSize)
    pos_test = 0

    for i in range(test_iteration_time):
        item_final_test.append(test[0][pos_test:pos_test + test_batchSize])
        item_final_test.append(test[1][pos_test:pos_test + test_batchSize])
        pos_test+=test_batchSize
        final_test.append(item_final_test)
        item_final_test=[]
    if test_size % test_batchSize!=0:
        item_final_test.append(test[0][pos_test:])
        item_final_test.append(test[1][pos_test:])
        final_test.append(item_final_test)

    # print(len(validation[0]))
    # print(len(final_validation))
    # print(len(final_validation[0]))
    # print(len(final_validation[0][0]))
    return training_size,validation,final_test

def next_batch(number):
    global training
    temp_training,r_image,r_label=[],[],[]

    for image,label in zip(training[0],training[1]):
        temp_training.append([image,label])

    np.random.shuffle(temp_training)
    temp_training=temp_training[:number]

    for item in temp_training:
        r_image.append(item[0])
        r_label.append(item[1])

    return r_image,r_label


# def normalizeCTValueByLungWindow(data):
#     tData = data
#     for i in range(len(tData[0])):
#         d = np.int32((tData[0][i] + 1350) / 1500.0 * 255.0)
#         d[d>255]=255
#         d[d<0]=0
#         tData[0][i] = d
#     return tData

if __name__ == '__main__':
    Initialization(10)
    # next_batch(100)