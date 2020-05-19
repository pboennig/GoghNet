import os
import numpy as np
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold

def load_images(data_dir='data'):
	""" 
	Loads the original and fake images.
	"""
	fakes_dir = os.path.join(data_dir, 'fakes')
	orig_dir = os.path.join(data_dir, 'orig')
	fakes, origs = [], []
	for fake_file in os.listdir(fakes_dir):
		fake = None
		with Image.open(os.path.join(fakes_dir, fake_file)) as image:
			fake = image.copy()

		orig_file = fake_file.split('_')[0] + '.jpg'
		orig = None
		with Image.open(os.path.join(orig_dir, orig_file)) as image:
			orig = image.copy()

		if fake.mode == 'RGB' and image.mode == 'RGB':
			fakes.append(fake)
			origs.append(orig)
	return fakes, origs

def vectorize_images(images, new_size):
	vectorized = []
	for image in images:
		resized = image.resize(new_size)
		vector = np.array(resized).flatten()
		vectorized.append(vector)
	return np.stack(vectorized)

def process_images(fakes, origs):
	"""
	Resizes each image to the size of the smallest image in the dataset
	to ensure uniformly sized inputs to the classifier and mean centers
	the data. Returns two N X D design matrices.
	"""
	min_width = min(image.width for image in (fakes + origs))
	min_height = min(image.height for image in (fakes + origs))
	fakes_matrix = vectorize_images(fakes, (min_width, min_height))
	origs_matrix = vectorize_images(origs, (min_width, min_height))
	
	mean_image = np.mean((fakes_matrix + origs_matrix) / 2, axis=0)
	fakes_matrix = fakes_matrix - mean_image
	origs_matrix = origs_matrix - mean_image
	return fakes_matrix, origs_matrix

def evaluate_SVM(X, y, n_splits=5):
	kf = KFold(n_splits=n_splits, shuffle=True)
	scores = []
	for i, (train_index, test_index) in enumerate(kf.split(X)):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		clf = LinearSVC()
		clf.fit(X_train, y_train)
		scores.append(clf.score(X_test, y_test))
	return sum(scores) / n_splits

def main():
	fakes, origs = load_images()
	fakes, origs = process_images(fakes, origs)
	X = np.concatenate((fakes, origs))
	y = np.array([1. for _ in fakes] + [0. for _ in origs])
	score = evaluate_SVM(X, y)
	print("Linear SVM achieved mean accuracy of {}".format(score))


if __name__=='__main__':
	main()