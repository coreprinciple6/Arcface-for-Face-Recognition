''' This script compares image names and creates pairs of matched and mismatched images'''
import os,glob
import random
import csv 

datapath = '/home/coreprinciple/work_station/ML_project/FR/face_recognition/bfc_data'
fields = ['first_image', 'second_image', 'is_same']
filename = "pairs.csv"
rows = []
cnt = 1

for f in glob.glob(f'{datapath}/*'):
    for g in glob.glob(f'{datapath}/*'):

        f_choice = random.choices(os.listdir(f),k=5)
        g_choice = random.choices(os.listdir(g),k=5)
        #product = ((i, j) for i in glob.glob(f'{f}/*.jpg') for j in glob.glob(f'{g}/*.jpg'))
        product = ((i, j) for i in f_choice for j in g_choice)
        for i, j in product:
            #cnt = cnt +1
            print(i,'   ', j)
            i_name = i.split('_')[0]
            j_name = j.split('_')[0]
            if i_name==j_name:
                is_same = True
            else:
                is_same = False
            temp = [i,j,is_same]
            rows.append(temp)
    print('done')


# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(rows)

#print(cnt)
