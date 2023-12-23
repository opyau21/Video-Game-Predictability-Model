import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def clean():
    master = pd.read_csv("Games.csv", sep=',').drop(columns=['Title'])
    x = master.iloc[:, :-1]
    y = master.iloc[:, -1]

    x_train, x_test, train_labels, test_labels = train_test_split(x, y, test_size=0.2, random_state=1824897142)

    scalar = StandardScaler()
    x_train_scaled = pd.DataFrame(scalar.fit_transform(x_train))
    x_train_scaled.to_csv('scaled_train.csv')
    train_labels.to_csv('train_labels.csv')

    x_test_scaled = pd.DataFrame(scalar.fit_transform(x_test))
    x_test_scaled.to_csv('scaled_test.csv')
    test_labels.to_csv('test_labels.csv')


