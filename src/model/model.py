from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC

VOCABULARY_PATH="dataset/fillers.txt"
POSITIVE_PATH="dataset/pos/text1.txt"
NEGATIVE_PATH="dataset/neg/text1.txt"


def read_file(file_path: str):
    lines = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        for line in file:
            lines.append(line.rstrip())
    return lines


vocabulary = read_file(VOCABULARY_PATH)


def fit():
    positive_data = read_file(POSITIVE_PATH)
    pos_labels = [1 for i in range(len(positive_data))] # without fillers
    negative_data = read_file(NEGATIVE_PATH)
    neg_labels = [0 for i in range(len(negative_data))] # with fillers

    X = positive_data + negative_data
    Y = pos_labels + neg_labels

    """ делим на тестовые и тренировочные данные """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    """ слова в числа """
    vectorizer = CountVectorizer(vocabulary=vocabulary, stop_words={'russian'}, lowercase=False)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train, y_train)
    return classifier, vectorizer


def fit_example():
    vocabulary = read_file(VOCABULARY_PATH)
    positive_data = read_file(POSITIVE_PATH)
    pos_labels = [1 for i in range(len(positive_data))] # without fillers
    negative_data = read_file(NEGATIVE_PATH)
    neg_labels = [0 for i in range(len(negative_data))] # with fillers

    X = positive_data + negative_data
    Y = pos_labels + neg_labels

    # делим на тестовые и тренировочные данные
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # слова в числа
    vectorizer = CountVectorizer(vocabulary=vocabulary, stop_words={'russian'}, lowercase=False)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    """clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Evaluate the model on the test data
    score = clf.score(X_test, y_test)
    print("Test score:", score)"""

    # обучение
    #classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    #classifier.fit(X_train, y_train)
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train, y_train)

    # предсказываем
    y_pred = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    ex = "Машинное обучение — класс методов искусственного интеллекта, \
характерной чертой которых является непрямое решение задачи, \
а обучение — за счет применения решений множество сходных задач"
    ex = vectorizer.transform([ex])
    indices = ex.indices
    yy = classifier.predict_proba(ex)
    filler_words_in_sentence = [vocabulary[i] for i in indices]
    print(yy)
    print("Filler words in sentence:", filler_words_in_sentence)

#fit_example()