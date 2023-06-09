# Import module yang akan digunakan
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Load dataset
dataset = pd.read_csv('dataset/gabungan.csv')

# Membagi data training dan testing
X = dataset.drop('kelas', axis=1)
Y = dataset['kelas']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)
Y_train

# Inisiasi jumlah k pada kNN
knn3 = KNeighborsClassifier(n_neighbors=3)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn7 = KNeighborsClassifier(n_neighbors=7)

# Train model menggunakan data training
knn3.fit(X_train, Y_train)

# Melakukan prediksi / klasifikasi
Y_pred = knn3.predict(X_test)

cf_matrix = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cf_matrix, annot=True)

print("Recall: ", round(recall_score(Y_test, Y_pred, pos_label='adidas')*100, 1), "%")
print("Precision: ", round(precision_score(
    Y_test, Y_pred, pos_label='adidas')*100, 1), "%")
print("F1-Score: ", round(f1_score(Y_test, Y_pred, pos_label='adidas')*100, 1), "%")
print("Accuracy:", round(metrics.accuracy_score(Y_test, Y_pred)*100, 1), "%")