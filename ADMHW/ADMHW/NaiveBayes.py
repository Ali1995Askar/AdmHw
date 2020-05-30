import os
from csv import reader
from math import sqrt
from math import exp
from math import pi
from collections import Counter

def fill_missing_data(rows, attr, most_common):
	for i in range(len(rows)):
		if rows[i][attr] == "?":
			rows[i][attr] = most_common

def load_csv(file_path, separate=","):
	with open(file_path, "r") as csv_file:
		csv_obj = reader(csv_file)
		dataset = [row for row in csv_obj]
		headers = dataset[0]
		rows = dataset[1:]

		rows_num = len(rows)

		name_to_index = {name: index for index, name in enumerate(headers)}

		attr_type_ids = {"categorical": set(),
						 "continuous": set()
						}

		# cast numeric data to float
		# mark the attribute as continuous/categorical
		for attr_name in headers:

			c_idx = name_to_index[attr_name]
			is_continuous = False
			attr_values = []

			for r_idx in range(rows_num):
				val = rows[r_idx][c_idx]

				numeric_type = val.isnumeric()
				# for checking that the whole column contains continuous data
				is_continuous |= numeric_type

				if numeric_type:
					rows[r_idx][c_idx] = float(val.strip())
				
				# used later for calculating the most label in a certain attribute
				attr_values.append(rows[r_idx][c_idx])

			# save indexes of each type of data continuous|categorical
			if is_continuous:
				attr_type_ids["continuous"].add(c_idx)
			else:
				attr_type_ids["categorical"].add(c_idx)

			# using "attr_values" to compute frequencies of each label of the attribute "attr_name"
			count = Counter(attr_values)
			fill_missing_data(rows, c_idx, count.most_common(1)[0][0])


		# train_set, test_set = split_data(rows, rows_num)

		return {
			"headers": headers,
			"dataset": rows,
			"attr_type_ids": attr_type_ids,
			"name_to_index": name_to_index
		}

class Bayes:
	def __init__(self, dataset, attr_type):
		self.dataset = dataset
		self.attr_type = attr_type
		self.mapper = self.__labels_mapper()
		# process the dataset
		self.__map_labels()
		# create counter for labels for each class of the target attribute 
		self.target_labels_counter = self.__calc_class_label_counter()
		# create model summaries
		self.summaries = self.__filter_processed_data_by_class()

	# separate the dataset by target values and calculate the labels frequencies
	def __calc_class_label_counter(self):
		filtered = self.__filter_data_by_class()
		labels_counter = {}
		for label, rows in filtered.items():
			labels_counter[label] = {}
			for idx, col in enumerate(zip(*rows)):
				if idx in self.attr_type["continuous"]:
					continue
				count = Counter(col)
				labels_counter[label][idx] = {key: val for key, val in count.items()}
		return labels_counter

	# create label:integer mapper
	def __labels_mapper(self):
		mapper = {}
		for idx, column in enumerate(zip(*self.dataset)):

			unique = list(set(column))
			unique.sort()
			mapper[idx] = {}
			for i, val in enumerate(unique):
				mapper[idx][val] = i if idx in self.attr_type["categorical"] else val
		
		return mapper

	# convert categorical values to numbers
	def __map_labels(self):
		for row in range(len(self.dataset)):
			for col in range(len(self.dataset[0])):
				self.dataset[row][col] = self.mapper[col][self.dataset[row][col]]

	# Split the dataset by class values, returns a dictionary
	def __filter_data_by_class(self):
		filtered = {}
		for i in range(len(self.dataset)):
			row = self.dataset[i]
			label = row[-1]
			if label not in filtered:
				filtered[label] = []
			filtered[label].append(row)
		return filtered

	# Calculate the mean of a list of numbers
	def __mean(self, num):
		return sum(num) / len(num)

	# Calculate the standard deviation of a list of numbers
	def __stdev(self, num):
		average = self.__mean(num)
		variance = sum([(x-average)**2 for x in num]) / (len(num) - 1)
		return sqrt(variance)

	# Calculate the mean, stdev and count for each column in a dataset
	def __process_dataset(self, rows):
		summaries = [(self.__mean(column), self.__stdev(column), len(column)) for column in zip(*rows)]
		# remove the summary of the target attribute
		return summaries[:-1]

	# Split dataset by class then calculate statistics for each row
	def __filter_processed_data_by_class(self):
		filtered = self.__filter_data_by_class()
		summaries = {}
		for label, rows in filtered.items():
			summaries[label] = self.__process_dataset(rows)
		return summaries

	# Calculate the Gaussian probability distribution function for x
	def __gaussian_density_func(self, x, mean, stdev):
		exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
		return (1 / (sqrt(2 * pi) * stdev)) * exponent

	# Calculate the probabilities of predicting each class for a given row
	def __calculate_class_probabilities(self, row):
		rows_num = sum([self.summaries[label][0][2] for label in self.summaries])
		probabilities = {}
		for label, label_summaries in self.summaries.items():
			probabilities[label] = self.summaries[label][0][2] / rows_num
			for i in range(len(label_summaries)):
				mean, stdev, size = label_summaries[i]

				if i in self.attr_type["continuous"]:
					probabilities[label] *= self.__gaussian_density_func(row[i], mean, stdev)
				else: 
					probabilities[label] *= self.target_labels_counter[label][i][row[i]] / size

		return probabilities

	# Predict the class for a given row
	def predict(self, row):
		prediction = None
		positive, negative = self.get_labels_probabilities(row)
		
		if positive > negative:
			prediction = "positive"
		else:
			prediction = "negative"

		return prediction


	def process_input(self, row):
		for col in range(len(row)):
			if col not in self.attr_type["categorical"]:
				continue
			row[col] = self.mapper[col][row[col]]

	def get_prob(self, row):
		prob_positive = self.__calculate_class_probabilities(row)[1]
		prob_negative = self.__calculate_class_probabilities(row)[0]
		
		positive = prob_positive / (prob_negative + prob_positive)
		negative = prob_negative / (prob_negative + prob_positive)

		return positive, negative

filename = os.path.join( os.path.dirname(__file__), "heart_disease_male.csv")
data = load_csv(filename)

bayes = Bayes(dataset=data["dataset"],
					attr_type=data["attr_type_ids"])

# # define a new record
# row = [43, "asympt", 140, "FALSE", "normal", 135, "yes", "positive"]
# bayes.process_input(row)
# # # predict the label
# label = bayes.predict(row)
# print(f"Data={row}, Predicted: {label}")
# # # print probabilities for each class
# positive, negative = bayes.get_labels_probabilities(row)
# print(f"Negative: {negative}")
# print(f"Positive: {positive} ")

