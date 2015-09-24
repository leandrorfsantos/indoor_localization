import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def training_kmeans(X_train,y_train,X_test,y_test):
	#X_train= preprocessing.normalize(X_train)
	#X_test = preprocessing.normalize(X_test)
	pca = PCA(n_components=None)
	X_train = pca.fit_transform(X_train)
	X_test = pca.transform(X_test)
	kmeans = KMeans(n_clusters=200)
	kmeans.fit(X_train,y_train)
	y_pred = kmeans.predict(X_test)
	print metrics.accuracy_score(y_test,y_pred)

def training_knn_simple(X_train,y_train,X_test,y_test):
	### knn
	#### best k~25... accuracy_score 0.39
	# train_test_split 0.4 best k = 14
	k_range = range(5,50)
	#pca = PCA(n_components=None)
	#X_train = pca.fit_transform(X_train)
	#X_test = pca.transform(X_test)
	scores=[]
	for k in k_range:
		knn = neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance')
		knn.fit(X_train, y_train)
		ypred = knn.predict(X_test)
		#scores.append(metrics.accuracy_score(y_test, ypred)) 
		print metrics.accuracy_score(y_test,ypred)

def training_svm_grid(X_train,y_train,X_test,y_test):
	### svm
	X_train_norm = preprocessing.normalize(X_train)
	X_test_norm = preprocessing.normalize(X_test)
	C_range = np.arange(0.1,5,0.3)
	gamma_range = np.arange(1,100,1)
	param_grid = dict(gamma=gamma_range.tolist(), C=C_range.tolist()) 
	#cv = StratifiedShuffleSplit(y_train, n_iter=5, test_size=0.2, random_state=42)
	grid = GridSearchCV(svm.SVC(), param_grid=param_grid,n_jobs=-1)
	print 'fiting...'
	grid.fit(X_train,y_train)
	print("The best classifier is: ", grid.best_estimator_)
	print("The best score: ", grid.best_score_)
	y_pred = grid.predict(X_test)
	print metrics.accuracy_score(y_test, y_pred)

def training_svm_parameters():
	## SVM
	g_range=np.arange(0.1,10.1,0.5)
	c_range=np.arange(0.1,3,0.2)
	g_scores=[]
	pca = PCA(n_components=None)
	X_pca_train = pca.fit_transform(X_train)
	X_pca_test = pca.transform(X_test)
	c_scores=[]
	for g in g_range:
		print g
		print 'running svm Accuracy Tests.'
		for c in c_range:
			print c
			rbf_svc = svm.SVC(kernel='rbf', gamma=g, C=c, shrinking=True)
			rbf_svc.fit(X_pca_train, y_train)
			#rbf_svc.fit(x_train_norm,y_train_trans)
			y_pred = rbf_svc.predict(X_pca_test)
			c_scores.append(metrics.accuracy_score(y_test, y_pred))
			#print metrics.classification_report(y_test_trans, ypred)
			#print metrics.accuracy_score(y_test_trans, ypred)
			#print metrics.confusion_matrix(y_test_trans, ypred)
		print c_scores			
		plt.plot(c_range, c_scores)
		plt.xlabel('Value of c, g fixed in ' + str(g))
		plt.ylabel('Accuracy test')
		plt.title('svm pca n_components: ' + str(n) + 'g='+str(g))
		plt.savefig('svm_pca_g_c_test'+str(n)+str(int(g*10))) 
		plt.close()

		c_scores = []

def training_svm_simple(X_train,y_train,X_test,y_test):
	X_train_norm = preprocessing.normalize(X_train)
	X_test_norm = preprocessing.normalize(X_test)
	#scaler = preprocessing.StandardScaler()
	#X_train = scaler.fit_transform(X_train)
	#clf = svm.SVC(kernel='rbf', gamma=1.5, C=5)
	clf = svm.SVC(C=5, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=1.5,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	print metrics.accuracy_score(y_test,y_pred)

def main():
	indoor_localization_train = pd.read_csv("./data/train.csv")
	indoor_localization_test = pd.read_csv("./data/test.csv")

	# shuffle the rows in the train csv
	indoor_localization_train_shuffled = indoor_localization_train.iloc[np.random.permutation(len(indoor_localization_train))]

	y_train = indoor_localization_train_shuffled.label
	X_train = indoor_localization_train_shuffled[
	['AP01', 'AP02', 'AP03', 'AP04', 'AP05', 'AP06', 'AP07', 'AP08', 'AP09', 'AP10', 'AP11', 
	'AP12', 'AP13', 'AP14', 'AP15', 'AP16', 'AP17', 'AP18', 'AP19']].fillna(0)

	X_test = indoor_localization_test[
	['AP01', 'AP02', 'AP03', 'AP04', 'AP05', 'AP06', 'AP07', 'AP08', 'AP09', 'AP10', 'AP11', 
	'AP12', 'AP13', 'AP14', 'AP15', 'AP16', 'AP17', 'AP18', 'AP19']].fillna(0)
	y_test = indoor_localization_test.label

	le = preprocessing.LabelEncoder()
	le.fit(y_test)
	y_test = le.transform(y_test)

	le.fit(y_train)
	y_train = le.transform(y_train)


	"""
	from sklearn.feature_selection import SelectKBest, f_regression
	from sklearn.pipeline import make_pipeline
	# ANOVA SVM-C
	anova_filter = SelectKBest(f_regression, k=5)
	clf = svm.SVC(kernel='rbf', gamma=1, C=100)
	X_train = preprocessing.normalize(X_train)
	X_test = preprocessing.normalize(X_test)
	anova_svm = make_pipeline(anova_filter, clf)
	anova_svm.fit(X_train, y_train)
	y_pred = anova_svm.predict(X_test)
	print metrics.accuracy_score(y_test,y_pred)
	"""

	#training_svm_grid(X_train,y_train,X_test,y_test)
	training_knn_simple(X_train,y_train,X_test,y_test)
	#training_knn_pca_parameters(X_train,y_train,X_test,y_test)
	#training_kmeans(X_train,y_train,X_test,y_test)
	#clf.fit(x,y)
	#joblib.dump(clf,'ds_hiring.pkl')
	#clf = joblib.load('ds_hiring.pkl')
if __name__ == "__main__":
  	main()
