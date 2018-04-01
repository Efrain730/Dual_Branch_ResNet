import os
import numpy as np

Training_period = True
Test_period = True

l1_training_dir = 'D:/PaperPreparation/CTEigenValue/train/ev1/'
l2_training_dir = 'D:/PaperPreparation/CTEigenValue/train/ev2/'
l3_training_dir = 'D:/PaperPreparation/CTEigenValue/train/ev3/'

l1_test_dir = 'D:/PaperPreparation/CTEigenValue/test/ev1/'
l2_test_dir = 'D:/PaperPreparation/CTEigenValue/test/ev2/'
l3_test_dir = 'D:/PaperPreparation/CTEigenValue/test/ev3/'

out_training_dir = 'D:/PaperPreparation/Eigen_Value/train/'
out_test_dir = 'D:/PaperPreparation/Eigen_Value/test/'

training_file_list = os.listdir(l1_training_dir)
test_file_list = os.listdir(l1_test_dir)

if os.path.isdir(out_training_dir) == False:
    os.makedirs(out_training_dir)
if os.path.isdir(out_test_dir) == False:
    os.makedirs(out_test_dir)

# def read():
#     for file_name in training_file_list:
#         print(file_name)
#         f = open(out_training_dir+file_name,'r')
#         for line in f:
#             line = line[:-2]
#             line = line.split(',')
#             print(len(line))

def gather_information(list1,list2,list3):
    result = []

    for i in range(len(list1)):
        result.append(list1[i])
        result.append(list2[i])
        result.append(list3[i])

    result_str=''
    for i in range(len(result)):
        result_str += (str(result[i])+',')
    return result_str

def str_to_list(str):
    str = str[:-2]
    str = str.split(',')
    for i in range(len(str)):
        str[i] = float(str[i])
    return str

def main():
    if Training_period:
        for file_name in training_file_list:
            f_l1 = open(l1_training_dir+file_name,'r')
            f_l2 = open(l2_training_dir+file_name,'r')
            f_l3 = open(l3_training_dir+file_name,'r')
            f_out = open(out_training_dir+file_name,'w')

            for line_l1, line_l2, line_l3 in zip(f_l1, f_l2, f_l3):
                if line_l1.index('\n')!=1:
                    f1_list = str_to_list(line_l1)
                    f2_list = str_to_list(line_l2)
                    f3_list = str_to_list(line_l3)

                    gather = gather_information(f1_list,f2_list,f3_list)
                    f_out.write(gather+'\n')
                else:
                    f_out.write(line_l1)
            f_out.close()

    if Test_period:
        for file_name in test_file_list:
            f_l1 = open(l1_test_dir + file_name, 'r')
            f_l2 = open(l2_test_dir + file_name, 'r')
            f_l3 = open(l3_test_dir + file_name, 'r')
            f_out = open(out_test_dir + file_name, 'w')

            for line_l1, line_l2, line_l3 in zip(f_l1, f_l2, f_l3):
                if line_l1.index('\n') != 1:
                    f1_list = str_to_list(line_l1)
                    f2_list = str_to_list(line_l2)
                    f3_list = str_to_list(line_l3)

                    gather = gather_information(f1_list, f2_list, f3_list)
                    f_out.write(gather + '\n')
                else:
                    f_out.write(line_l1)
            f_out.close()

if __name__ == '__main__':
    main()