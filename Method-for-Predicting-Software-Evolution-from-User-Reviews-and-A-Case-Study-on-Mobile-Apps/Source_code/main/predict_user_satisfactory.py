# -*- coding:utf-8 -*-
'''
Created on 2018.12.11

@author: Molly Song
'''


import os
import re
import sys
import numpy
import pandas
import gensim
from collections import Counter, defaultdict
from util_tools import calculate_phrase_vector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


PSOURCE_DIR = r'D:\programming\workspacePycharm\masterProject\Data\DataForPrediction'
MODEL_PATH = r'D:\programming\workspacePycharm\masterProject\Word2VectorModel\wiki.en.text.vector'
MODEL = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)


def get_whatsnew_feature(category, app_name):
    app_dir = os.path.join(PSOURCE_DIR, category, app_name)
    valid_update_days = None
    whatsnew_contnets = defaultdict(list)
    for feature in os.listdir(app_dir):
        one_file = open(os.path.join(app_dir, feature, 'WhatsNew.txt'), 'r')
        lines = one_file.readlines()
        one_file.close()

        if not valid_update_days:
            update_days = list(map(lambda x: int(x.split(':::')[0]), lines))
            valid_update_days = remove_successive_days(update_days)

        for line in lines:
            day, update_flag = list(map(int, line.split(':::')))
            if update_flag and day in valid_update_days:
                whatsnew_contnets[day].append(feature)

    pairs = sorted(whatsnew_contnets.items())
    content_vectors = []
    for _, whatsnew_contnet in pairs:
        word_vectors = []
        for keyword in whatsnew_contnet:
            strange_word = False
            for word in keyword.split():
                try:
                    word = re.sub(':', '', word)
                    word_vectors.append(MODEL[word])
                except KeyError:
                    strange_word = True
                    break
            if strange_word:
                continue
        content_vector = calculate_phrase_vector(word_vectors)
        content_vectors.append(content_vector)

    return content_vectors


def get_data(data_dir, content_vectors, threshold):
    vaild_features_path = os.path.join(data_dir, 'valid_features.xlsx')
    empty_flag = True
    ys = []
    xs = []
    content_len = len(content_vectors)
    postfixs = ['freq', 'pos', 'neg']
    print('\tThreshold={}, Number of filtered features:'.format(threshold), end=' ')
    for postfix in postfixs:
        content_index = 0
        filter_file_path = os.path.join(data_dir, 'feature_kendall_%s.xlsx' % postfix)
        if not os.path.exists(filter_file_path):
            xs.append([])
            continue

        df_kendall = pandas.read_excel(filter_file_path)
        df_filtered_kendall = df_kendall[abs(df_kendall.kendall_correlation_coefficient) > threshold]
        filtered_feature_names = df_filtered_kendall.index.tolist()
        filtered_feature_count = len(filtered_feature_names)
        if filtered_feature_count == 0:
            xs.append([])
            continue

        empty_flag = False
        x = []
        filtered_features = pandas.read_excel(vaild_features_path)[filtered_feature_names]
        for _, row in filtered_features.iterrows():
            one_property = row.tolist()
            one_property.extend(list(content_vectors[content_index % content_len]))
            x.append(one_property)
            content_index += 1

        xs.append(x)

        df_labels = pandas.read_excel(os.path.join(data_dir, 'labels_%s.xlsx' % postfix))
        ys.append(df_labels['label'].tolist())

        print('{}'.format(filtered_feature_count), end=' ')
    print(' ')
    return empty_flag, xs, ys


MIN_PERIOD = 2


def remove_successive_days(update_days):
    num_days = len(update_days)
    valid_update_days = [update_days[0]]
    for i in range(num_days):
        if i - 1 >= 0 and update_days[i] - update_days[i - 1] <= MIN_PERIOD:
            continue
        valid_update_days.append(update_days[i])
    return valid_update_days


def predict_user_satisfactory_factor(x, y):
    clf = LogisticRegression()
    scores = cross_val_score(clf, numpy.matrix(x), numpy.array(y), scoring='accuracy', cv=5)
    return scores


def round2_str(a):
    if a is None:
        return '0.00'
    return str(round(a, 2))


THRESHOLD_EFFECT_DIR = r'D:\programming\workspacePycharm\masterProject\Data\Predict_result\Threshold_effect'


def predict_with_varying_threshold(category, app_name):
    content_vectors = get_whatsnew_feature(category, app_name)
    app_data_dir = os.path.join(DATA_DIR, category, app_name, 'Update_satisfactory')
    avg_scores = [[], [], []]
    valid_thresholds = [[], [], []]
    threshold = 1.0
    while threshold >= -0.05:
        empty_flag, xs, ys = get_data(app_data_dir, content_vectors, threshold)
        if empty_flag:
            threshold -= 0.05
            continue

        index = 0
        for x, y in zip(xs, ys):
            if len(set(y)) <= 1:
                index += 1
                continue
            try:
                _scores = predict_user_satisfactory_factor(x, y)
            except BaseException:
                index += 1
                continue
            _average_score = numpy.mean(_scores)
            avg_scores[index].append(_average_score)
            valid_thresholds[index].append(threshold)
            index += 1

        threshold -= 0.05

    save_dir = os.path.join(THRESHOLD_EFFECT_DIR, app_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    postfixes = ['freq', 'pos', 'neg']
    for one_valid_thresholds, one_avg_scores, post_fix in zip(valid_thresholds, avg_scores, postfixes):
        f = open(os.path.join(save_dir, 'US_{}.txt'.format(post_fix)), 'w')
        for t, s in zip(one_valid_thresholds, one_avg_scores):
            f.write('{} {}\n'.format(t, s))
        f.close()


PREDICT_RESULT_DIR = r'D:\programming\workspacePycharm\masterProject\Data\Predict_result'


def get_classificaton_detail(category, app_name):
    task_names = ['Frequency prediction',
                  'Positive sentiment score prediction',
                  'Negative sentiment score prediction']

    content_vectors = get_whatsnew_feature(category, app_name)
    app_data_dir = os.path.join(DATA_DIR, category, app_name, 'Update_satisfactory')
    predict_result_file = open(os.path.join(PREDICT_RESULT_DIR, category, app_name, 'US.txt'), 'r')
    line = predict_result_file.readline()
    predict_result_file.close()
    thresholds = list(map(float, line.strip('\n').split()))

    for i in range(3):
        empty_flag, xs, ys = get_data(app_data_dir, content_vectors, thresholds[i])
        all_test = []
        all_predict = []
        for j in range(5):
            x_train, x_test, y_train, y_test = train_test_split(xs[i], ys[i], test_size=0.2)
            clf = LogisticRegression()
            clf.fit(x_train, y_train)
            y_predict = clf.predict(x_test)
            all_test.extend(y_test)
            all_predict.extend(y_predict)

        print(task_names[i])
        print(classification_report(all_test, all_predict))


DATA_DIR = r'D:\programming\workspacePycharm\masterProject\analyse_correlation\Data'


if __name__ == '__main__':
    category = 'Travel & Local'
    app_name = 'com_yelp_android'
    predict_with_varying_threshold(category, app_name)

    '''
    category_app_pairs = [['Music & Audio', 'com_smule_singandroid'],
                          ['Travel & Local', 'com_yelp_android'],
                          ['Social', 'com_snapchat_android'],
                          ['Photography', 'com_google_android_apps_photos'],
                          ['Shopping', 'com_amazon_mShop_android_shopping']]
    for category, app_name in category_app_pairs:
        print(app_name)
        get_classificaton_detail(category, app_name)
    '''
    '''

    category_path = r'D:\programming\workspacePycharm\masterProject\AppCategory'
    predict_result_path = r'D:\programming\workspacePycharm\masterProject\Data\Predict_result'
    # for category in ['Books & Reference', 'Business', 'Education',
    #                  'Social', 'Communication', 'Finance', 'Maps & Navigation',
    #                  'News & Magazines', 'Travel & Local']:
    # for category in ['Entertainment', 'Games']:
    for category in ['Shopping']:
        file_name = '{}.txt'.format(category)
    # for file_name in os.listdir(category_path):
    #     category = file_name.split('.')[0]
        print(category)
        f = open(os.path.join(category_path, file_name), 'r')
        app_names = f.readlines()
        f.close()
        for app_name in app_names:
            app_name = app_name.strip('\n')
            print(app_name)
            app_data_dir = os.path.join(DATA_DIR, category, app_name, 'Update_satisfactory')
            if not os.path.exists(app_data_dir):
                continue

            print('Start generating update content vectors')
            content_vectors = get_whatsnew_feature(category, app_name)

            print('Start getting data')
            threshold = 1.0
            counter = [None, None, None]
            scores = [None, None, None]
            worse_counts = [0, 0, 0]
            average_score = [0, 0, 0]
            best_threshold = [0, 0, 0]
            while threshold >= 0.1:
                empty_flag, xs, ys = get_data(app_data_dir, content_vectors, threshold)
                if empty_flag:
                    threshold -= 0.05
                    continue

                index = 0
                for x, y in zip(xs, ys):
                    if len(set(y)) <= 1 or worse_counts[index] >= 3:
                        index += 1
                        continue
                    try:
                        _scores = predict_user_satisfactory_factor(x, y)
                    except BaseException:
                        index += 1
                        continue
                    _average_score = numpy.mean(_scores)
                    if _average_score > average_score[index]:
                        average_score[index] = _average_score
                        scores[index] = _scores
                        counter[index] = Counter(y)
                        best_threshold[index] = threshold
                        worse_counts[index] = 0
                    else:
                        worse_counts[index] += 1
                    index += 1

                if worse_counts[0] >= 3 and worse_counts[1] >= 3 and worse_counts[2] >= 3:
                    break

                threshold -= 0.05

            task_names = ['Frequency prediction',
                          'Positive sentiment score prediction',
                          'Negative sentiment score prediction']

            for i in range(len(task_names)):
                if scores[i] is None:
                    continue
                print('\t{}'.format(task_names[i]))

                num_y = sum(counter[i].values())
                for k, v in counter[i].items():
                    print('\t%d : %f(%d/%d)' % (k, float(v) / num_y, v, num_y), end='')
                    # predict_result_file.write('%d : %f(%d/%d)' % (k, float(v) / num_y, v, num_y))

                print('\n\t\tBest Threshold: %f' % best_threshold[i])
                print('\t\tAll Accuracies:')
                for s in scores[i]:
                    print('\t\t\t%s' % s)
                print('\t\tAverage Accuracy: %f' % average_score[i])

            predict_dir = os.path.join(predict_result_path, category, app_name)
            if not os.path.exists(predict_dir):
                os.makedirs(predict_dir)
            predict_result_file = open(os.path.join(predict_dir, 'US.txt'), 'w')
            predict_result_file.write('{}\n'.format(' '.join(list(map(round2_str, best_threshold)))))
            predict_result_file.write('{}\n'.format(' '.join(list(map(str, average_score)))))
            predict_result_file.close()
    '''