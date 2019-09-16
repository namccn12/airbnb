import pandas as pd
import math
import re
from collections import OrderedDict

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
             "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
             "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
             "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
             "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
             "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
             "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
             "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
             "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


def remove_special_char_n_stopwords(text):
    without_special_char_text = re.sub(r"[^a-z]+", " ", text.lower())
    without_special_char_text_list = without_special_char_text.split(" ")
    result = []
    for word in without_special_char_text_list:
        if word not in stopwords:
            result.append(word)
    return result


def generate_word_dict(unique_words, bag_of_words):
    word_dict = dict.fromkeys(unique_words, 0)
    for word in bag_of_words:
        word_dict[word] += 1
    return word_dict


def compute_TF(word_dict, bag_of_words):
    tf_dict = {}
    bag_of_words_count = len(bag_of_words)
    for word, count in word_dict.items():
        tf_dict[word] = count / float(bag_of_words_count)
    return tf_dict


def compute_IDF(documents):
    n = len(documents)

    id_dict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                id_dict[word] += 1

    for word, val in id_dict.items():
        id_dict[word] = math.log(n / float(val))
    return id_dict


def compute_TFIDF(tf, idf_matrix):
    tfidf = {}
    for word, val in tf.items():
        tfidf[word] = val * idf_matrix[word]
    return tfidf


####################
# start of program #
####################

data = pd.read_csv("AB_NYC_2019.csv")

# remove outliers and select specific neighbourhood_group
df = data[(data.price <= 600) & (data.neighbourhood_group == "Manhattan")].copy()

reviews_group_1 = ""
reviews_group_2 = ""
reviews_group_3 = ""

# classify all records into 3 groups based on number_of_reviews
# reviews_group_1: less than 20 reviews
# reviews_group_2: between 20 and 100 reviews
# reviews_group_3: more than 100 reviews
for index, row in df.iterrows():
    if row["number_of_reviews"] < 20:
        reviews_group_1 = reviews_group_1 + " " + str(row["name"])
    elif 20 <= row["number_of_reviews"] < 100:
        reviews_group_2 = reviews_group_2 + " " + str(row["name"])
    else:
        reviews_group_3 = reviews_group_3 + " " + str(row["name"])

# remove special characters and stopwords
# then split by space
bag_of_words_1 = remove_special_char_n_stopwords(reviews_group_1)
bag_of_words_2 = remove_special_char_n_stopwords(reviews_group_2)
bag_of_words_3 = remove_special_char_n_stopwords(reviews_group_3)

# get set of unique words across all groups
unique_words = set(bag_of_words_1).union(set(bag_of_words_2), set(bag_of_words_3))

# create a word:count dictionary for each group
word_dict_1 = generate_word_dict(unique_words, bag_of_words_1)
word_dict_2 = generate_word_dict(unique_words, bag_of_words_2)
word_dict_3 = generate_word_dict(unique_words, bag_of_words_3)

# calculate TF score for each group based on word:count dictionary
tf_1 = compute_TF(word_dict_1, bag_of_words_1)
tf_2 = compute_TF(word_dict_2, bag_of_words_2)
tf_3 = compute_TF(word_dict_3, bag_of_words_3)

# calculate IDF score for all words
idf = compute_IDF([word_dict_1, word_dict_2, word_dict_3])

# calculate TF-IDF score for a specific group
tf_idf = compute_TFIDF(tf_3, idf)

# print top 10 the most important words
dictionary = (pd.DataFrame([tf_idf]).T.to_dict()[0])
ordered_dict = OrderedDict(sorted(dictionary.items(), key=lambda kv: kv[1], reverse=True))

for i in list(ordered_dict)[:10]:
    print(i)
