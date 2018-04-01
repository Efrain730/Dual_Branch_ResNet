import os
import numpy as np

# pre_file_dir='D:/PaperPreparation/CTEigenValue/train/ct/'
# file_dir='D:/PaperPreparation/processed_result_avg_var/train/'

pre_file_dir='D:/PaperPreparation/CTEigenValue/test/ct/'
file_dir='D:/PaperPreparation/processed_result_avg_var/test/'

if not os.path.isdir(file_dir):
    os.makedirs(file_dir)

file_names=os.listdir(pre_file_dir)
for file in file_names:
    line_counter=0
    image=[]
    tmp_image=[]
    label=[]

    pre_f=open(pre_file_dir+file,'r')
    f=open(file_dir+file,'w')

    for line in pre_f:
        if line.index('\n')==1:
            label.append(int(line))
        else:
            string=line[:-2]
            split_str=string.split(',')
            for pixel in split_str:
                tmp_image.append(float(pixel))
            line_counter+=1
            if line_counter % 32==0:
                image.append(tmp_image)
                tmp_image=[]

    image -= np.mean(image,axis=0)
    image /= np.std(image,axis=0)


    for single_image,single_label in zip(image,label):
        counter=0
        for pixel in single_image:
            f.write(str(pixel)+',')
            counter += 1
            if counter % 32 == 0:
                f.write('\n')
        f.write(str(single_label)+'\n')
