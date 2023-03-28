from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

VOCABULARY_PATH="dataset/fillers.txt"
POSITIVE_PATH="dataset/pos/text1.txt"
NEGATIVE_PATH="dataset/neg/text1.txt"


def read_file(file_path: str):
    lines = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        for line in file:
            lines.append(line)
    return lines


"""
 movie_data = load_files(r"D:\txt_sentoken") 
 X, y = movie_data.data, movie_data.target 
"""
def fit():
    vocabulary = read_file(VOCABULARY_PATH)
    positive_data = read_file(POSITIVE_PATH)
    pos_labels = [1 for i in range(len(positive_data))]
    negative_data = read_file(NEGATIVE_PATH)
    neg_labels = [0 for i in range(len(negative_data))]

    X = positive_data + negative_data
    Y = pos_labels + neg_labels

    # делим на тестовые и тренировочные данные
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # слова в числа
    vectorizer = CountVectorizer(vocabulary=vocabulary, stop_words={'russian'})
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    """clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Evaluate the model on the test data
    score = clf.score(X_test, y_test)
    print("Test score:", score)"""

    # обучение
    classifier = RandomForestClassifier(n_estimators=500, random_state=0)
    classifier.fit(X_train, y_train)

    # предсказываем
    y_pred = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

fit()