import os
import operator
import random
# import argparses

# training_path = "../lung_data_mean_std/train/"
# test_path="../lung_data_mean_std/test/"

training_path = "../PaperPreparation/processed_result_avg_var/train/"
test_path="../PaperPreparation/processed_result_avg_var/test/"
lamda_training_path = '../PaperPreparation/Eigen_Value/train/'
lamda_test_path = '../PaperPreparation/Eigen_Value/test/'

# training_path = "../lung_data/train/"
# test_path="../lung_data/test/"

def extractFiles(path, training=False, test=False):
    if training:
        print('Training Data dir:',training_path)
    if test:
        print('Test Data dir:',test_path)

    file_Names = os.listdir(path)
    temp_File_Names = []
    for f in file_Names:
        temp_File_Names.append(f[:-8])
    opacity_class = sorted(list(set(temp_File_Names)))

    filesInClasses = []
    for c in opacity_class:
        # print(c)
        ind = getStrIndex(temp_File_Names, c)
        # print(ind)
        filesInClasses.append(random.sample([file_Names[i] for i in ind],len(ind)))
        # print(filesInClasses)
    # for f in filesInClasses:
    #     print(f)
    return filesInClasses


def getStrIndex(strList, str):
    ind = []
    for i in range(len(strList)):
        if operator.eq(strList[i],str)==1:
            ind.append(i)
    return ind


if __name__ == '__main__':
    extractFiles(training_path)
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--path", required = True, help = "input path")
    # args = vars(ap.parse_args())
    # extractFiles(args["path"])

