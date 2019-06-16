import csv
import numpy as np
jump_first_line = True
csv_reader = csv.reader(open('fer2013.csv', 'r'))

train_num = [0 for i in range(7)]
val_num = [0 for i in range(7)]
test_num = [0 for i in range(7)]

for item in csv_reader:
    if(jump_first_line):
        jump_first_line = False
        continue
    if(item[2] == 'Training'):
        train_num[int(item[0])] += 1
    elif item[2] == 'PublicTest':
        val_num[int(item[0])] += 1
    elif item[2] == 'PrivateTest':
        test_num[int(item[0])] += 1
print('train:', train_num, 'total:{}'.format(np.sum(train_num)))
print('val:', val_num,'total:{}'.format(np.sum(val_num)))
print('test', test_num, 'total:{}'.format(np.sum(test_num)))


# reader = csv.reader(open('label_val.csv', 'r'))
# writer = csv.writer(open('label_train_val.csv', 'a', newline=''), dialect='excel')

# for item in reader:
#     writer.writerow(item)