def data_reader(object):
	def __init__(self, img_dir, log_dir):
		self.train_start_pointer = 0
		self.valid_start_pointer = 0
		self.img_dir = img_dir
		self.log_dir = log_dir
		self.images = []
		self.steerings = []

	def train_generator(self, batch_size):



	def valid_generator(self, batch_size):



	def read_log(self, smoothing = False):
		steerings = []
		center = []
		left = []
		right = []
		speed = []
	with open(self.log_dir) as f:
		lines = f.read().split("\n")
		for i in range(len(lines)-2):
			line = lines[i+1]
			items = line.split(",")
			steerings.append(items[3])
			center.append(items[0])
			left.append(items[1])
			right.append(items[2])
			speed.append(items[-1])



	def cameral_adjust(self, right):


	def read_img(self, img_dir):



	def shuffle(self):

	def augments(self):

