# -*- coding:utf-8 -*-
'''
Created on 2018.01.26

@author: Molly Song
'''

import numpy
import random
from scipy.interpolate import interp1d


def interpolate_data(standard_period, freq_list, pos_list, neg_list):
    _freq_list = freq_list[:]
    _pos_list = pos_list[:]
    _neg_list = neg_list[:]
    days = range(1, len(_freq_list) + 1)
    funInterp_f = interp1d(days, _freq_list)    # linear interpolation when not enough property: 27
    funInterp_p = interp1d(days, _pos_list)    # linear interpolation when not enough property: 27
    funInterp_n = interp1d(days, _neg_list)    # linear interpolation when not enough property: 27
    while len(_freq_list) < standard_period:
        chosenDay = random.choice(days[:-1])
        chosenIndex = days.index(chosenDay)
        dayNew = (chosenDay + days[chosenIndex + 1])/2.0    # between two consecutive days
        days.insert(chosenIndex + 1, dayNew)
        _freq_list.insert(chosenIndex + 1, funInterp_f(dayNew))
        _pos_list.insert(chosenIndex + 1, funInterp_p(dayNew))
        _neg_list.insert(chosenIndex + 1, funInterp_n(dayNew))
    return _freq_list, _pos_list, _neg_list


def sample_data(standard_period, freq_list, pos_list, neg_list):
    # for frequency
    tmp_freq = freq_list[:]
    _freq_list = tmp_freq[0:1]
    dict_f = dict(enumerate(freq_list))
    # for positive score
    tmp_pos = pos_list[:]
    _pos_list = tmp_pos[0:1]
    dic_p = dict(enumerate(pos_list))
    # for negative score
    tmp_neg = neg_list[:]
    _neg_list = tmp_neg[0:1]
    dic_n = dict(enumerate(neg_list))
    # do sampling
    randomList = random.sample(range(len(freq_list)), standard_period - 2)
    randomList.sort()
    _freq_list.extend([dict_f[i] for i in randomList])
    _freq_list.append(tmp_freq[-1])
    _pos_list.extend([dic_p[i] for i in randomList])
    _pos_list.append(tmp_pos[-1])
    _neg_list.extend([dic_n[i] for i in randomList])
    _neg_list.append(tmp_neg[-1])
    return _freq_list, _pos_list, _neg_list


def get_train_n_test_data_with_detail(x_all, y_all, num_reviews, num_days_before_update, ratio):
    data_len = len(y_all)
    test_index = random.sample(range(data_len), data_len / ratio)
    test_x = []
    test_y = []
    test_num_reviews = []
    test_day_before_update = []
    train_x = []
    train_y = []
    for i in range(data_len):
        if i in test_index:
            test_x.append(x_all[i])
            test_y.append(y_all[i])
            test_num_reviews.append(num_reviews[i])
            test_day_before_update.append(num_days_before_update[i])
        else:
            train_x.append(x_all[i])
            train_y.append(y_all[i])
    return train_x, train_y, test_x, test_y, test_num_reviews, test_day_before_update


def get_lower_n_upper_bound(level_range):
    # print(level_range)
    range_split = level_range.split(',')
    lower_bound = float(range_split[0].strip('('))
    upper_bound = float(range_split[1].strip(']'))
    return lower_bound, upper_bound


def record_update_prediction_detail(cluster_name, p_model, x, y, y_detail, num_reviews, num_days_before_update, ratio, file_path):
    f_R = open(file_path,'a')
    f_R.write("%s\n" % cluster_name)
    p_matrix = [[0 for i in range(5)] for j in range(5)]
    for t in range(5):
        tr_x, tr_y, te_x, te_y, te_num_rev, te_days = get_train_n_test_data_with_detail(x, y, num_reviews, num_days_before_update, ratio)
        p_model.fit(numpy.matrix(tr_x), numpy.array(tr_y))
        right = 0
        for m, n, o, p in zip(numpy.matrix(te_x), numpy.array(te_y), te_num_rev, te_days):
            pre_n = p_model.predict(m)
            p_matrix[int(n)][int(pre_n)] += 1    # for calculate the accuracy of every urgency level's prediction
            l, u = get_lower_n_upper_bound(str(y_detail[int(pre_n)]))
            l_r = l * o
            u_r = u * o
            f_R.write("    Actual : %d(%d)\n" % (n, p))
            f_R.write("    Predict : %d(%f to %f)\n" % (pre_n, l_r, u_r))
            f_R.write("        Detail : %f = %f x %d, %f = %f x %d \n" % (l_r, l, o, u_r, u, o))
            if pre_n == n:
                right += 1
        f_R.write("Accuracy : %f\n" % (float(right) / len(te_y)))
    f_R.write("Detail:\n")
    for i in range(5):
        if sum(p_matrix[i]) != 0:
            f_R.write("%d %f\n" % (i, float(p_matrix[i][i]) / sum(p_matrix[i])))
        else:
            f_R.write("%d 2.0\n" % i)
    f_R.close()
    return


def calculate_phrase_vector(vector_list):
    return numpy.nan_to_num(numpy.array(vector_list).sum(axis=0) / len(vector_list))

