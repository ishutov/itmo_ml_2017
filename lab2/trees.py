import numpy
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# разделение датасета на тестовую и обучающую выборку
def split_dataset(size):
	ds = pandas.read_csv('seeds_dataset.txt', sep='\t', lineterminator='\n', header=None).values
	ds_attributes = ds[:, :-1] # атрибуты семени
	ds_class = ds[:, -1].astype(numpy.int64, copy=False) # класс семени
	return train_test_split(ds_attributes, ds_class, test_size=size, random_state=55)

def main():
	max_size = 0.4
	min_size = 0.1
	step = 0.1
	for size in numpy.arange(min_size, max_size, step):
		data_train, data_test, class_train, class_test = split_dataset(size)
		
		decisionForest = DecisionTreeClassifier() 
		decisionForest = decisionForest.fit(data_train, class_train)
		decisionAcc = decisionForest.score(data_test, class_test)
		
		randonForest = RandomForestClassifier()
		randonForest = randonForest.fit(data_train, class_train)
		randomAcc = randonForest.score(data_test, class_test)
		
		print('Size: ', size)
		print('Decision Tree accuracy: ', round(decisionAcc,10))
		print('Random Tree accuracy: ', round(randomAcc,10))
		print('\n')

main()