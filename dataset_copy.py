import os
import shutil
trainset = './AIZOO/train/'
valset = './AIZOO/val/'
trainpath = './train'
valpath = './val'
if not os.path.exists(trainpath):
    os.makedirs(trainpath + '/Annotations')
    os.makedirs(trainpath + '/JPEGImages')
if not os.path.exists(valpath):
    os.makedirs(valpath + '/Annotations')
    os.makedirs(valpath + '/JPEGImages')
i=0
j=0
f = open('./train/train.txt', 'w')
for file in sorted(os.listdir(trainset)):
    if 'test' in file and i < 800:
        i = i + 1
        if os.path.splitext(file)[1] == '.xml':
            print(file)
            r = shutil.copy(trainset + file, os.path.join('./train/Annotations/'))
            print('copy path is ' + r)
        elif os.path.splitext(file)[1] == '.jpg':
            print(file)
            r = shutil.copy(trainset + file, os.path.join('./train/JPEGImages/'))
            print('copy path is ' + r)
            f.write(str(file) + '\n')
            print("write image in txt")
    if 'test' not in file and j < 800:
        j = j + 1
        if os.path.splitext(file)[1] == '.xml':
            print(file)
            r = shutil.copy(trainset + file, os.path.join('./train/Annotations/'))
            print('copy path is ' + r)
        elif os.path.splitext(file)[1] == '.jpg':
            print(file)
            r = shutil.copy(trainset + file, os.path.join('./train/JPEGImages/'))
            print('copy path is ' + r)
            f.write(str(file) + '\n')
            print("write image in txt")
f.close()

i=0
j=0
f = open('./val/val.txt', 'w')
for file in sorted(os.listdir(valset)):
    if 'test' in file and i <500:
        i=i+1
        if os.path.splitext(file)[1] == '.xml':
                print(file)
                r=shutil.copy(valset + file,os.path.join('./val/Annotations/'))
                print('copy path is '+ r)
        elif os.path.splitext(file)[1] == '.jpg':
                print(file)
                r=shutil.copy(valset + file,os.path.join('./val/JPEGImages/'))
                print('copy path is '+ r)
                f.write(str(file)+'\n')
                print("write image in txt")
    if 'test'  not in file and j <500:
        j = j + 1
        if os.path.splitext(file)[1] == '.xml':
            print(file)
            r = shutil.copy(valset + file, os.path.join('./val/Annotations/'))
            print('copy path is ' + r)
        elif os.path.splitext(file)[1] == '.jpg':
            print(file)
            r = shutil.copy(valset + file, os.path.join('./val/JPEGImages/'))
            print('copy path is ' + r)
            f.write(str(file) + '\n')
            print("write image in txt")
f.close()