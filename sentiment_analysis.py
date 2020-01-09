import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
import argparse

#remove punctuation and stopwords, make reviews lower-case and apply lemma.
#also apply stemming and lemmatization
def preprocess_reviews(reviews, stem,lemm):
    words_only = re.compile("[.;:!\'?,\"()\[\]]")
    reviews = [words_only.sub("", line.lower()) for line in reviews]
    #remove stop words
    stop_words = set(stopwords.words('english'))
    removed_stop_words = []
    for review in reviews:
        tmp=' '.join([word for word in review.split() if word not in stop_words])
        removed_stop_words.append(tmp)
    #apply stemmping
    if stem:
        stemmer = PorterStemmer()
        reviews =[' '.join([stemmer.stem(word) for word in review.split()]) for review in removed_stop_words]

    #Apply Lemmatization
    if lemm:
        lemmatizer = WordNetLemmatizer()
        reviews = [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in removed_stop_words]

    return reviews

def top_features(X,Y,X_valid,y_valid):
    max=0
    best_num_of_features=10000
    for n in np.arange(10000, 100000, 10000):
        ch2 = SelectKBest(chi2, k=n)
        x_train_chi2 = ch2.fit_transform(X, Y)
        x_validation_chi2 = ch2.transform(X_valid)
        clf = LinearSVC(C=.1)
        clf.fit(x_train_chi2, Y)
        score = clf.score(x_validation_chi2, y_valid)
        if score>max:
            max=score
            best_num_of_features=n
        #print("chi2 feature selection evaluation calculated for {} features".format(n), score)

    return best_num_of_features
def initiate_list(trainPath,testPath,devPath):
    reviews_train_pos = []
    train_pos=trainPath+'/imdb_train_pos.txt'
    for line in open(train_pos, 'r'):
        reviews_train_pos.append(line.strip())
    reviews_train_neg = []
    train_neg = trainPath + '/imdb_train_neg.txt'
    for line in open(train_neg, 'r'):
        reviews_train_neg.append(line.strip())

    reviews_test_pos = []

    test_pos = testPath + '/imdb_test_pos.txt'
    for line in open(test_pos, 'r'):
        reviews_test_pos.append(line.strip())
    reviews_test_neg = []
    test_neg = testPath + '/imdb_test_neg.txt'
    for line in open(test_neg, 'r'):
        reviews_test_neg.append(line.strip())

    reviews_dev_pos = []
    dev_pos = devPath + '/imdb_dev_pos.txt'
    for line in open(dev_pos, 'r'):
        reviews_dev_pos.append(line.strip())
    reviews_dev_neg = []
    dev_neg = devPath + '/imdb_dev_neg.txt'
    for line in open(dev_neg, 'r'):
        reviews_dev_pos.append(line.strip())
    return(reviews_train_pos,reviews_train_neg,reviews_test_pos,reviews_test_neg,reviews_dev_pos,reviews_dev_neg)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-train-file', nargs='+', type=str, help="specify the directory of the of the training set.")
    parser.add_argument('--input-test-file', nargs='+', type=str, help="specify the directory of the of the testing set.")
    parser.add_argument('--input-dev-file', nargs='+', type=str, help="specify the directory of the of the development set.")

    args = parser.parse_args()
    print(args)
    try:
        s=args.accumulate(args.integers)
        print(s)
    except Exception as e:
        pass

    default_train="datasets/IMDb/train/"
    if args.input_train_file is not None:
        default_train = args.input_train_file[0]

    default_test = "datasets/IMDb/test/"
    if args.input_test_file is not None:
        default_test = args.input_test_file[0]
    default_dev = "datasets/IMDb/dev/"
    if args.input_dev_file is not None:
        default_dev = args.input_dev_file[0]

    #return the reviews in a list form
    reviews_train_pos, reviews_train_neg, reviews_test_pos, reviews_test_neg, reviews_dev_pos, reviews_dev_neg = initiate_list(default_train,default_test,default_dev)

    #data cleaning
    reviews_train=preprocess_reviews(reviews_train_pos,stem=False,lemm=False)
    len_train_pos=len(reviews_train)
    reviews_train_neg=preprocess_reviews(reviews_train_neg,stem=False,lemm=False)
    reviews_train.extend(reviews_train_neg)

    reviews_test=preprocess_reviews(reviews_test_pos,stem=False,lemm=False)
    len_test_pos = len(reviews_test)
    reviews_test_neg=preprocess_reviews(reviews_test_neg,stem=False,lemm=False)
    reviews_test.extend(reviews_test_neg)

    reviews_dev=preprocess_reviews(reviews_dev_pos,stem=False,lemm=False)
    len_dev_pos = len(reviews_dev)
    reviews_dev_neg=preprocess_reviews(reviews_dev_neg,stem=False,lemm=False)
    reviews_dev.extend(reviews_dev_neg)

    # 1 means positive review, whereas 0 means negative review
    target_train = [1 for i in range(len_train_pos)]
    target_neg_train=[0 for j in range(len_train_pos, len(reviews_train))]
    target_train.extend(target_neg_train)

    # 1 means positive review, whereas 0 means negative review
    target_test = [1 for i in range(len_test_pos)]
    target_neg_test=[0 for j in range(len_test_pos, len(reviews_test))]
    target_test.extend(target_neg_test)

    # 1 means positive review, whereas 0 means negative review
    target_dev = [1 for i in range(len_dev_pos)]
    target_neg_dev_train=[0 for j in range(len_dev_pos, len(reviews_dev))]
    target_dev.extend(target_neg_dev_train)

    #vectorization, change each sentence into zero's and one's (e.g., one if a word exist in our corpus)
    #without word_count
    #cv = CountVectorizer(binary=True)
    #without word_count and bigram
    #cv = CountVectorizer(binary=True, ngram_range=(1, 2))
    #without wordcount
    #cv = CountVectorizer(binary=False)
    #without wordcount and ngram
    #cv = CountVectorizer(binary=False,ngram_range=(1, 2))
    #TF-IDF
    #cv = TfidfVectorizer()
    cv = TfidfVectorizer(binary=True, ngram_range=(1, 2))
    cv.fit(reviews_train)
    X_train = cv.transform(reviews_train)
    X_test = cv.transform(reviews_test)

    X_dev_test=cv.transform(reviews_dev)

    find_best_k=top_features(X_train,target_train,X_dev_test,target_dev)
    #print(find_best_k)
    ch2 = SelectKBest(chi2, k=find_best_k)

    x_train_chi2 = ch2.fit_transform(X_train, target_train)
    x_validation_chi2_selected = ch2.transform(X_test)
    clf = LinearSVC(C=.1)

    lr_output=clf.fit(x_train_chi2, target_train)

    yhat_classes = lr_output.predict(x_validation_chi2_selected)
    accuracy = accuracy_score(target_test, yhat_classes)
    precision = precision_score(target_test, yhat_classes)
    recall = recall_score(target_test, yhat_classes)
    f1 = f1_score(target_test, yhat_classes)
    print("Precision=%s : Recall=%s : FScore=%s : Accuracy=%s" % (precision,recall,f1,accuracy))
