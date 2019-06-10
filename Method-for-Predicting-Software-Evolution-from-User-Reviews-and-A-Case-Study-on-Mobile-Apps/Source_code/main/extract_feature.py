# -*- coding:utf-8 -*-
'''
Created on 2017.08.18

@author: Molly Song
'''
import codecs
import re
import os
import nltk
import copy
import string
import itertools
import collections
from pyltp import Segmentor
from collections import defaultdict
from textblob import Word, TextBlob


def load_deling_stop_words(file_path):
    f = open(file_path, 'r')
    words = [w.strip('\n') for w in f.readlines()]
    f.close()
    return words


def load_spliting_stop_words(file_path):
    f = open(file_path, 'r')
    words = [w.strip('\n') for w in f.readlines()]
    f.close()
    return words


def get_whatsnew_day_n_text(matches):
    for item in matches:
        if item:
            return item.group('days'), item.group('text')
    print("NO MATCH")
    exit(0)


def strip_colon(word_tag):  # remove colon, because it's a special mark
    return re.sub(':', '', word_tag[0]), word_tag[1]


RE_NUMERIC = re.compile(r"[0-9]+")


def strip_numeric(word_tag):
    return (RE_NUMERIC.sub("", word_tag[0]), word_tag[1])


WN = nltk.stem.WordNetLemmatizer()


# DEL_STOPWORDS = ["between","about","against","each","more","same","out","where"]
# ADD_STOPWORDS = ["ever", "please", "app", "apps","need", "im", "u", "also","could","would","might","must","need","since","cuz","bcoz","coz"]
# STOP_WORDS = list(set(nltk.corpus.stopwords.words("english"))-set(DEL_STOPWORDS) | set(ADD_STOPWORDS))
def lemmatize_word(word_tag):
    if word_tag[0]:
        return WN.lemmatize(word_tag[0]), word_tag[1]
    return "", word_tag[1]


STOP_WORDS_DEL = load_deling_stop_words(
    r'D:\programming\workspacePycharm\masterProject\Data\English_stopwords_delete.txt')


def remove_stop_words(word_tag):
    if word_tag[0] in STOP_WORDS_DEL:
        return "", word_tag[1]
    else:
        return word_tag


FILTERS = [strip_colon, strip_numeric, lemmatize_word, remove_stop_words]


def apply_filters(word_tag):
    for f in FILTERS:
        word_tag = f(word_tag)
    return word_tag


def filter_word(words):
    return map(apply_filters, nltk.pos_tag(words))


POINT_MARK = re.compile(r"^\W+$")


def is_invalid_sentence(words):
    if len([w for w in words if not POINT_MARK.match(w)]) > len(words) / 2:
        return False
    else:
        # for w in words:
        #    print("%s " % w),
        # print"\n"
        return True


def is_english(words):
    count = 0.0
    for word in words:
        #         w = Word(word)
        # spell check and reduce the effect of stemming
        if Word(word).spellcheck()[0][1] != 1:
            '''
            and Word(word + 's').spellcheck()[0][1] != 1 and Word(word + 'e').spellcheck()[0][1] != 1 and Word(word[:-2]).spellcheck()[0][1] != 1
            '''
            count += 1
        elif len(word) == 2:  # all word with length 2 will be judged as correct by spellcheck
            count += 0.5
    if count >= len(words) / 2 + 1:
        #         print "DEL: ",
        #         print words
        return False
    return True


def clean_text(text):
    segmentor = Segmentor()
    segmentor.load(r'D:\programming\ltp_model_3.4.0\ltp_data\cws.model')
    sentences = nltk.sent_tokenize(  # benefit sentence tokenizing
        text.lower())  # lowercase
    cleaned_text_tag = []
    for sentence in sentences:
        words = segmentor.segment(sentence)
        # words = nltk.word_tokenize(sentence)
        if is_invalid_sentence(words):
            continue
        cleaned_sentence_tag = [cs for cs in filter_word(words) if cs[0]]
        cleaned_text_tag.extend(cleaned_sentence_tag)
    cleaned_text_tag.append(".")
    segmentor.release()
    return cleaned_text_tag


'''
DEL_PUNTS = ['\'', '-']
PUNCTS = list(set(string.punctuation)-set(DEL_PUNTS))
'''
ABBREVIATION = re.compile(r"^\w+'\w+$")
STOP_WORDS_SPL = load_spliting_stop_words(
    r'D:\programming\workspacePycharm\masterProject\Data\English_stopwords_split.txt')
PUNCTS = list(string.punctuation)
DELIMITER = set(STOP_WORDS_SPL + PUNCTS)
VAILD_TAG = ["NN", "VB"]


def get_phrase(text_tag):
    _text = copy.copy(text_tag)
    #     phrases_tag = [item[1] for item in itertools.groupby(_text, lambda x:
    #                                                                      x[0] in DELIMITER or
    #                                                                      POINT_MARK.match(x[0]) or
    #                                                                      ABBREVIATION.match(x[0]))
    #                                                                       if not item[0]]
    phrase_tag_groups = itertools.groupby(_text, lambda x: x[0] in DELIMITER or POINT_MARK.match(x[0]) or ABBREVIATION.match(x[0]))
    phrases = []
    for group in phrase_tag_groups:
        if not group[0]:
            one_phrase = []
            for word_tag in list(group[1]):
                if len(word_tag[0]) > 1 or word_tag[1][:2] in VAILD_TAG:
                    one_phrase.append(word_tag[0])
            if one_phrase:
                phrases.append(tuple(one_phrase))

    text_tag_group = itertools.groupby(text_tag, lambda x: x in DELIMITER)
    stripped_text = []
    for group in text_tag_group:
        one_group = []
        for word_tag in list(group[1]):
            one_group.append(word_tag[0])
        if one_group:
            stripped_text.append(tuple(one_group))

    #     stripped_text = [list(item[1]) for item in itertools.groupby(text_tag, lambda x: x in DELIMITER)]
    return stripped_text, phrases  # phrase:tuple of words


def get_word_freq_dic(all_words):
    word_freq_dic = defaultdict(lambda: 0)
    for word in all_words:  # count word
        word_freq_dic[word] += 1
    return word_freq_dic


def get_word_degree_dic(co_occurance_graph):
    word_degree_dic = defaultdict(lambda: 0)
    for key in co_occurance_graph:  # calculate degree
        word_degree_dic[key] = sum(co_occurance_graph[key].values())
    return word_degree_dic


def get_word_score_dic(word_freq_dic, word_degree_dic):
    #     if len(word_freq_dic) != len(word_degree_dic):
    #         print "WRONG! len(word_freq_dic) != len(word_degree_dic)"
    #         print ("len(word_freq_dic) = %d " % len(word_freq_dic))
    #         print ("len(word_degree_dic) = %d " % len(word_degree_dic))
    #         exit(0)
    word_score_dic = {}
    for key in word_degree_dic:
        if key in word_freq_dic:
            word_score_dic[key] = 1.0 * word_degree_dic[key] / word_freq_dic[key]
        else:
            print("WRONG! word_freq_dic doesn't have key : ",)
            print(key)
    return word_score_dic


KEYWORD_NUM_THRESHOLD = 240


def get_keywords(phrase_freq_dic, word_score_dic):
    keywords = []
    for key in phrase_freq_dic:
        score = 0.0
        for word in key:
            score += word_score_dic[word]
        keywords.append((key, score, phrase_freq_dic[key]))
    #     keywords.sort(reverse = True, key = lambda x : x[1])
    #     len_keywords = len(keywords)
    #     if len_keywords <= KEYWORD_NUM_THRESHOLD:
    #         return {item[0]:item[2] for item in keywords}
    #     else:
    #         return {item[0]:item[2] for item in keywords[: len_keywords / 3]}

    return {item[0]: item[2] for item in keywords}


COUNT_THRESHOLD = 2
LEN_THRESHOLD = 7
MIN_COUNT = -10 ** 7


def combine_keywords(keywords, split_stripped_text, phrase_senti_score):
    first_word = False  # whether the first word is not stopword
    pre_stop = False  # whether last word is a stopword
    pre_group = None  # last stop word, if present word is not stopword, pre_group can be added to one, bc ther cannot be continuous stopwords in one keywords
    to_delete = []
    combined_keywords = defaultdict(lambda: 0)
    count = 0
    one = tuple()
    all_delete = []
    comb_keyword_list = []
    for group in split_stripped_text:
        if group in keywords:
            if pre_group:
                one += pre_group
                comb_keyword_list.append(pre_group)
                count += 1
            one += group
            comb_keyword_list.append(group)
            count += 1
            to_delete.append(group)
            first_word = True  # first word is not stopword
            pre_stop = False
            pre_group = None
        elif first_word and not pre_stop and len(group) < 2 and group[0] in STOP_WORDS_SPL:
            pre_group = group
            pre_stop = True
        else:  # get one combined keyword
            if count > 1 and len(one) <= LEN_THRESHOLD:  # more than one sub keyword
                combined_keywords[one] += 1  # count
                com_keyword_count = combined_keywords[one]
                if com_keyword_count >= COUNT_THRESHOLD:  # over threshold
                    all_delete.extend(to_delete)
                    score_set = set([])
                    get_one = False
                    for subk in comb_keyword_list:
                        #                         print(subk)
                        proper_subk_score = [k for k, v in collections.Counter(phrase_senti_score[subk]).items() if
                                             v >= com_keyword_count]
                        if get_one:
                            score_set = score_set & proper_subk_score
                        else:
                            score_set = proper_subk_score
                    if len(score_set) > 0:
                        phrase_senti_score[one].extend(list(score_set))
                    else:
                        combined_keywords[one] = MIN_COUNT
            to_delete = []
            one = tuple()
            first_word = False
            pre_stop = False
            pre_group = None
            count = 0
            comb_keyword_list = []
    for ad in set(all_delete):
        del keywords[ad]
    for key in combined_keywords:
        if combined_keywords[key] >= COUNT_THRESHOLD:
            #             print(key)
            keywords[key] = combined_keywords[key]
    return


def keywords(text, if_rev):
    split_stripped_text = []
    phrase_freq_dic = defaultdict(lambda: 0)
    co_occurance_graph = defaultdict(lambda: defaultdict(lambda: 0))
    all_key_words = []
    phrase_senti_score = defaultdict(lambda: [])
    for one in text:  # one review/update
        if if_rev:  # sentiment score for review
            sen_blob = TextBlob(one)
            senti_score = sen_blob.sentiment.polarity
        else:  # sentiment score for what's new
            senti_score = 0
        phrase_from_dialog = []
        doub_quo_split = one.split('"')
        if len(doub_quo_split) > 2:
            dialog_txt = doub_quo_split[1::2]
            phrase_from_dialog = [tuple(txt.lower().split()) for txt in dialog_txt]
            user_words = ','.join(doub_quo_split[::2])
        else:
            user_words = ' '.join(doub_quo_split)
        cleaned_user_words_tag = clean_text(user_words)
        stripped_sntns, phrase_in_sntns = get_phrase(cleaned_user_words_tag)
        phrase_in_sntns.extend(phrase_from_dialog)
        split_stripped_text.extend(stripped_sntns)
        all_key_words.extend(itertools.chain.from_iterable(phrase_in_sntns))
        for pis in phrase_in_sntns:
            if is_english(pis):
                phrase_senti_score[pis].append(senti_score)
                phrase_freq_dic[pis] += 1  # count phrase
                for (word, coword) in itertools.product(pis, pis):  # prepare for word degree calculation
                    co_occurance_graph[word][coword] += 1
    word_freq_dic = get_word_freq_dic(all_key_words)  # get word frequency
    word_degree_dic = get_word_degree_dic(co_occurance_graph)  # get word degree
    word_score_dic = get_word_score_dic(word_freq_dic, word_degree_dic)  # get word score
    keywords = get_keywords(phrase_freq_dic, word_score_dic)  # calculate keywords score and get first 1/3
    combine_keywords(keywords, split_stripped_text, phrase_senti_score)
    return keywords, phrase_senti_score


def write_to_file(path, keywords_with_count, phrase_senti_score):
    all_count = sum(keywords_with_count[key] for key in keywords_with_count) * 1.0
    keywords_with_freq = [(key, keywords_with_count[key] / all_count) for key in
                          keywords_with_count]  # calculate frequency
    #     keywords_with_freq = [(" ".join(list(key)), keywords_with_count[key]) for key in keywords_with_count]
    keywords_with_freq.sort(reverse=True, key=lambda x: (x[1], x[0]))  # sort
    #     with codecs.open(path,'a', encoding='utf-8') as f:
    #         for one in keywords_with_freq:    # write to file
    #             f.write("%s:%s\n" % (one[0], one[1]))
    f = open(path, "a", encoding='utf-8', errors='ignore')
    for one in keywords_with_freq:  # write to file
        keyword_tup = one[0]
        f.write("%s:::%s:::%s\n" % (" ".join(list(keyword_tup)), one[1], " ".join(map(str, phrase_senti_score[keyword_tup]))))
    f.close()

    '''# PRINT OUT FOR TEST
    for one in keywords_with_freq:    # write to file
        print("%s:%s\n" % (one[0], one[1]))
    '''
    return


def get_review_keywords_and_write_to_file(one_day, review_one_day, f_path, if_rev):
    keywords_with_count, phrase_senti_score = keywords(review_one_day, if_rev)  # extract
    #     print(len(keywords_with_count))
    write_to_file(os.path.join(f_path, '%d.txt' % one_day), keywords_with_count, phrase_senti_score)  # write
    return


PATH_KEYWORDS = r"D:\programming\workspacePycharm\masterProject\Data\Keywords"


def keywords_by_day(folder_path, if_rev):
    f_dir = os.path.join(PATH_KEYWORDS, '\\'.join(folder_path.split("\\")[-2:]))
    if not os.path.exists(f_dir):
        os.makedirs(f_dir)  # create directory
    for file_name in os.listdir(folder_path):
        timeDay = int(file_name.split(".")[0])
        print(timeDay)
        one_day_data_file = open(os.path.join(folder_path, file_name), 'r', encoding='utf-8', errors='ignore')
        one_day_data = one_day_data_file.readlines()
        one_day_data_file.close()
        get_review_keywords_and_write_to_file(timeDay, one_day_data, f_dir, if_rev)
    return


def extract_all_app_keywords(dir_path, if_rev):
    for f_name in os.listdir(dir_path):
        print(f_name)
        keywords_by_day(os.path.join(dir_path, f_name), if_rev)


'''
def keywords_hu (file_path, dest_path):
    with codecs.open(file_path,  encoding='utf-8') as a_file:
        lines = a_file.readlines()
        a_file.close()
    get_review_keywords_and_write_to_file(1, lines, dest_path)    # for the last day
'''

if __name__ == '__main__':
    ## FOR TEST
    # keywords_with_count, phrase_senti_score = keywords(["call improvements", "network improvements"], False)    # extract
    # print "RESULT:"
    # print keywords_with_count
    # write_to_file("", keywords_with_count, 0)    # write
    #######################################
    #
    # extract_all_app_keywords(r'D:\programming\workspacePycharm\masterProject\Data\OrderedSource\Review', True)
    extract_all_app_keywords(r'D:\programming\workspacePycharm\masterProject\Data\OrderedSource\Whatsnew', False)
