import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from sklearn.model_selection import train_test_split
import json
import h5py

dir_path = "./data/"


# data class
class data(object):
	def __init__(self, log_file_name, batch_size, test_ratio):
		self.dataset = []
		# self.train = []
		self.center = []
		self.left = []
		self.right = []
		self.speed = []
		self.steerings = []
		self.center, self.left, self.right, self.steerings = read_log(log_file_name)
		self.center, self.left, self.right, self.steerings = filt(self.center, self.left, self.right, self.steerings,0.001)
		self.dataset = generate_dataset(self.center, self.left, self.right, self.steerings)
		self.dataset = add_flip(self.dataset)
		self.dataset = set_filter(self.dataset)
		self.test_ratio = test_ratio
		self.train_dataset, self.valid_dataset = dataset_split(self.dataset,self.test_ratio)
		print("Length of training dataset is {},  Length of valid dataset is {}".format(len(self.train_dataset),len(self.valid_dataset)))
		self.next_train = train_generator(self.train_dataset, batch_size)
		self.next_valid = valid_generator(self.valid_dataset, batch_size)

# train valid set split
def dataset_split(all_set, ratio):
	X_train, X_valid = train_test_split(all_set, test_size = ratio)
	return X_train, X_valid

# read iamge name and angles from log file
def read_log(file_name):
	prev_prev = 0
	prev = 0
	center = []
	left = []
	right = []
	steerings = []
	with open(dir_path+file_name) as f:
		lines = f.read().split("\n")
		print("total input data of angles {}".format(len(lines)))
		for i in range(len(lines)-2):
			line = lines[i+1]
			items = line.split(", ")
			# using exponential smoothing
			steerings.append(0.90*float(items[3])+0.10*prev+0.0*prev_prev)
			prev_prev = prev
			prev = float(items[3])
			center.append(items[0])
			left.append(items[1])
			right.append(items[2])
	return center, left, right, steerings

# filter, randomly  drop some portion of angles below threshold
def filt(cen, le,ri,ste, threshold=0.001):
	delete_list = []
	for i in range(len(cen)):
		if ste[i] < threshold:
			if random.random() < 0.20:
				delete_list.append(i)
	for j in reversed(delete_list):
		del cen[j]
		del le[j]
		del ri[j]
		del ste[j]
	print("Number of steerings after filtering {}".format(len(ste)))
	return cen, le, ri, ste

# sample out some protion and large angles set, there are usually 3 peeks
def set_filter(dataset):
	angles = [x[2] for x in dataset]
	intervals = plt.hist(angles, 30)
	max_angle = np.max(intervals[0])
	index = [x for x in range(len(intervals[0])) if intervals[0][x] > max_angle*0.60]
	del_range = [(intervals[1][index[0]], intervals[1][index[0]+1]), (intervals[1][index[1]], intervals[1][index[1]+1]),
				(intervals[1][index[2]], intervals[1][index[2]+1])]
	dataset = filter_mods(dataset, del_range)
	print("total number of dataset after sampling is {}".format(len(dataset)))
	return dataset

# train generator
def train_generator(train_dataset, batch_size):
	length = len(train_dataset)
	random.shuffle(train_dataset)
	train_pointer = 0
	while True:
		y_batch = []
		x_batch = []
		while(len(y_batch)<batch_size):
			data = train_dataset[train_pointer%length]
			file = data[0]
			file_path = dir_path+file
			angle = data[2]
			img = read_img(file_path, data[1])
			# add conitrast and brightness
			if random.random() < 0.85:
				img = change_bright(img)
				if random.random() < 0.5:
					gamma = random.random()*5+0.2
					img = adjust_gamma(img, gamma)
			# random shift
			if random.random()<0.30:
				if random.random() < 0.80:
					# small shift
					img, angle = augment(img, angle, 12)
				else:
					# larger shift
					img, angle = augment(img, angle, 30)
			img = cv2.resize(img, (200, 66)) # resize image to fit into network
			img = img/255.0
			x_batch.append(img)
			y_batch.append(angle)
			train_pointer+=1
		x_batch_out = np.array(x_batch)
		y_batch_out = np.array(y_batch)
		yield x_batch_out, y_batch_out

# validation generator
def valid_generator(valid_dataset, batch_size):
	length = len(valid_dataset)
	random.shuffle(valid_dataset)
	valid_pointer = 0
	while True:
		y_batch = []
		x_batch = []
		while(len(y_batch) < batch_size):
			data = valid_dataset[valid_pointer%length]
			file = data[0]
			file_path = dir_path+file
			angle = data[2]
			img = read_img(file_path, data[1])
			img = cv2.resize(img, (200, 66))
			img = img/255.0
			x_batch.append(img)
			y_batch.append(angle)
			valid_pointer+=1
		x_batch_out = np.array(x_batch)
		y_batch_out = np.array(y_batch)
		yield x_batch_out, y_batch_out

# generate original dataset from log file
def generate_dataset(cen,le,ri,steer):
	dataset = []
	for i in range(len(cen)):
		data = (cen[i],False,steer[i])
		dataset.append(data)
	# left camera add angle 0.15
	for i in range(len(le)):
		ste = steer[i]+0.15
		data = (le[i], False, ste)
		dataset.append(data)
	# right camera subtract 0.15
	for i in range(len(ri)):
		ste = steer[i]-0.15
		data = (ri[i], False, ste)
		dataset.append(data)
	return dataset

# add flip images
def add_flip(dataset):
	new_data = []
	for item in dataset:
		# steering angles is negative origin
		new_data.append((item[0], True, -item[2]))
	dataset.extend(new_data)
	print("new dataset add filps, total number {}".format(len(dataset)))
	return dataset

# contrast adjustment
def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

# accurate camera adjustment(not used)
def camera_adjust(steering, right, speed):
	reaction_time = 2.0
	shift = 20.0 
	# mile/hour to feet/sec
	speed  = speed/3600.0*5280
	adjacent = speed*reaction_time*15.0 
	angle_adj = np.arctan(float(shift)/adjacent)
	if not right:
		angle_adj = - angle_adj
	steering = steering+angle_adj
	return steering

# read images and cropped
def read_img(img_file, flip):
	img = cv2.imread(img_file)
	img_crop = img[50:150,:,:]
	if flip:
		img_crop = cv2.flip(img_crop, 1)
	return img_crop

# select portion out of a dataset
def filter_mods(dataset, del_range):
	del_list = []
	for i in range(len(dataset)):
		data = dataset[i]
		if (data[2]>del_range[0][0] and data[2]<del_range[0][1]) or (data[2]>del_range[2][0] and data[2]<del_range[2][1]):
			a = random.random()
			if a < 0.55:
				del_list.append(i)
		if (data[2]>del_range[1][0]-0.005 and data[2]<del_range[1][1]-0.005):
			b = random.random()
			if b < 0.55:
				del_list.append(i)
	#     print(del_list)
	for j in reversed(del_list):
		del dataset[j]
	return dataset

# change brightness
def change_bright(img):
	image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	change = 0.20+np.random.uniform()*1.3
	image[:,:,2] = image[:,:,2]*change
	image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
	return image

# random shift images and its steering angles
def augment(img,steering, x_range):
	rows,cols,channles = img.shape
	dx = random.choice(range(-x_range,x_range))
	y_range = 12
	dy = random.choice(range(-y_range, y_range))
	d_angle = 0.0040*dx
	new_steering = steering+d_angle
	M = np.float32([[1,0,dx],[0,1,dy]])
	# affine tranformation
	new_img = cv2.warpAffine(img, M, (cols,rows))
	return new_img, new_steering