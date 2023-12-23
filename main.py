from Clean import *
from classification import *
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # clean()
    #
    # x_train = pd.read_csv('Data/scaled_train.csv').iloc[:, 1:]
    x_test = pd.read_csv('Data/scaled_test.csv').iloc[:, 1:]
    #
    # y_train = pd.read_csv('Data/train_labels.csv').iloc[:, 1:]
    y_test = pd.read_csv('Data/test_labels.csv').iloc[:, 1:]
    #
    #Logistic_Regression(x_train, y_train, x_test, y_test)

    # Predict with any game:
    model = tf.keras.models.load_model('Oliver Model.keras')
    # print(model.evaluate(x_test,y_test))

    scalar = StandardScaler()
    pred = pd.read_csv('Data/Prediction_Set.csv').iloc[:,1:]
    pred_scaled = pd.DataFrame(scalar.fit_transform(pred))


    print(model.predict(pred))

    # print(pd.DataFrame(np.round(model.predict(x_test))))
    # print(y_test)