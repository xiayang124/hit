from operator import truediv

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

INDIAN_TARGET_NAMES = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
                       'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                       'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                       'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                       'Stone-Steel-Towers']

PAVIA_UNIVERSITY_NAMES = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted_metal_sheets', 'Bare_Soil', 'Bitumen',
                          'Self_Blocking_Bricks', 'Shadows']

HOUSTION_NAMES = ['Unclassified', 'Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees', 'Soil', 'Water',
                  'Residential', 'Commercial', 'Road',
                  'Highway', 'Railway', 'Parking Lot 1', 'Parking Lot 2', 'Tennis Court', 'Running Track']

HONGHU_NAMES = ['Unclassified', 'Red roof', 'Road', 'Bare soil', 'Cotton', 'Cotton firewood', 'Rape', 'Chinese cabbage',
                'Pakchoi', 'Cabbage', 'Tuber mustard', 'Brassica parachinensis', 'Brassica chinensis',
                'Small Brassica chinensis', 'Lactuca sativa', 'Celtuce', 'Film covered lecttuce', 'Romaine lettuce',
                'Carrot', 'White radish', 'Garlic sprout', 'Broad bean', 'Tree']


class HSIEvaluation(object):
    def __init__(self, dataset_name) -> None:
        self.target_names = None
        if dataset_name == 'Indian':
            self.target_names = INDIAN_TARGET_NAMES
        elif dataset_name == "Pavia":
            self.target_names = PAVIA_UNIVERSITY_NAMES
        elif dataset_name == "Honghu":
            self.target_names = HONGHU_NAMES
        self.res = {}

    def AA_andEachClassAccuracy(self, confusion_matrix):
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        return each_acc, average_acc

    def eval(self, y_test, y_pred_test):
        class_num = np.max(y_test)
        classification = classification_report(y_test, y_pred_test,
                                               labels=list(range(class_num)), digits=4, target_names=self.target_names)
        oa = accuracy_score(y_test, y_pred_test)
        confusion = confusion_matrix(y_test, y_pred_test)
        each_acc, aa = self.AA_andEachClassAccuracy(confusion)
        kappa = cohen_kappa_score(y_test, y_pred_test)

        self.res['classification'] = str(classification)
        self.res['oa'] = oa * 100
        self.res['confusion'] = str(confusion)
        self.res['each_acc'] = str(each_acc * 100)
        self.res['aa'] = aa * 100
        self.res['kappa'] = kappa * 100
        return self.res
