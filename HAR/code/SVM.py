from sklearn import preprocessing
from sklearn.svm import SVC
from utils.Preprocessing import *
from utils.Plot import *
from utils.constants import *

prepare_data = Preprocessing()

train_X, train_y = prepare_data.statistics("train")
test_X,test_y = prepare_data.statistics("test")

# train_X = preprocessing.scale(train_X)
# test_X = preprocessing.scale(test_X)

clf = SVC(gamma='auto')
clf.fit(train_X, np.ravel(train_y))
pred_y = clf.predict(test_X)

heatmap(test_y, pred_y, "SVM"+str(NUM_FEATURES_USED))