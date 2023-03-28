from random import shuffle

src_path = '/home/shiyanshi/pingyi/dataset/smthv2/videofolder/train_videofolder.txt'
dst_path = '/home/shiyanshi/pingyi/dataset/smthv2/videofolder/train.txt'

src = open(src_path,'r')
dst = open(dst_path,'w')
train_data = src.readlines()
shuffle(train_data)
for data in train_data:
    dst.write(data)

dst.close()
src.close()