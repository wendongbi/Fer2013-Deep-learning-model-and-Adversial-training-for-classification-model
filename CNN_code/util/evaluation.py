import pandas as pd
import os
import shutil


class Evaluation(object):
    def __init__(self, gt_file, pred_file, thresh, img_dir=None):
        gt_temp = open(gt_file, 'r')
        pred_temp = open(pred_file, 'r')
        self.gt = {}
        self.pred = {}
        self.gt_count = [0 for i in range(31)]
        self.pred_count = [0 for i in range(31)]
        self.correct = [0 for i in range(31)]
        if img_dir is not None:
            os.mkdir('./WrongData')
            for i in range(31):
                os.mkdir('./WrongData/' + str(i))
        for line in gt_temp.readlines():
            name, label = line.split(',')
            self.gt[name] = int(label)
            self.gt_count[int(label)] += 1
        for line in pred_temp.readlines():
            name, label, confidence = line.split(',')
            if float(confidence) < thresh:
                continue
            self.pred_count[int(label)] += 1
            self.pred[name] = int(label)
        for (name, pred) in self.pred.items():
            if self.gt[name] == pred:
                self.correct[pred] += 1
            elif img_dir is not None:
                shutil.copy(
                    os.path.join(img_dir, name),
                    './WrongData/' + str(pred) + '/' + name)
        correct_sum = 0
        pred_sum = 0
        gt_sum = 0
        for i in range(30):
            correct_sum += self.correct[i]
            gt_sum += self.gt_count[i]
            pred_sum += self.pred_count[i]
        self.precision = correct_sum * 1.0 / pred_sum
        self.recall = correct_sum * 1.0 / gt_sum
        print('precision = ' + str(self.precision))
        print('recall = ' + str(self.recall))


# result = Evaluation(
#     gt_file='./classification/Annotations/test.csv',
#     pred_file='./classification/result.csv',
#     thresh=0.90)
# print('precision = ' + str(result.precision) + '\n')
# print('recall = ' + str(result.recall) + '\n')