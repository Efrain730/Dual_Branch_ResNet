import os
import numpy as np

pre_file_dir='D:/PaperPreparation/patch/test/'
file_dir='D:/Data/processed_result_CT/test/'

file_names=os.listdir(pre_file_dir)
for file in file_names:
    line_counter=0
    temp_image=[]

    pre_f=open(pre_file_dir+file,'r')
    f=open(file_dir+file,'w')

    for line in pre_f:
        if line.index('\n')==1:
            f.write(line)
        else:
            string=line[:-2]
            split_str=string.split(',')
            for pixel in split_str:
                temp_image.append(int(pixel))
            line_counter+=1
            if line_counter % 32==0:
                for i in range(len(temp_image)):
                    temp_image[i]=int((temp_image[i]+1350.0)/1500.0*255.0)
                temp_image=np.reshape(temp_image,[32,32]).tolist()
                for item in temp_image:
                    for number in item:
                        f.write(str(number)+',')
                    f.write('\n')
                temp_image=[]
    pre_f.close()
    f.close()