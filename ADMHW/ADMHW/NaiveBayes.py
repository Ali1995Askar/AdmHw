# Make Predictions with Naive Bayes On The Iris Dataset
from csv import reader
from math import sqrt
from math import exp
from math import pi
import os
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

class NaiveBayes:
	def __init__(self, dataset, attr_type_ids):
		self.dataset = dataset
		self.attr_type_ids = attr_type_ids
		self.lookup = self.__create_lookup_table()
		# process the dataset
		self.__label_to_number()
		# create counter for labels for each class of the target attribute 
		self.target_labels_counter = self.__create_target_labels_counter()
		# create model summaries
		self.model = self.__summarize_by_class()

	# separate the dataset by target values and calculate the labels frequencies
	def __create_target_labels_counter(self):
		separated = self.__separate_by_class()
		labels_counter = {}
		for class_value, rows in separated.items():
			labels_counter[class_value] = {}
			for idx, col in enumerate(zip(*rows)):
				if idx in self.attr_type_ids["continuous"]:
					continue
				count = Counter(col)
				labels_counter[class_value][idx] = {key: val for key, val in count.items()}
		return labels_counter

	# create label:integer mapper
	def __create_lookup_table(self):
		lookup = {}
		for idx, column in enumerate(zip(*self.dataset)):

			unique_values = list(set(column))
			unique_values.sort()
			lookup[idx] = {}
			for i, val in enumerate(unique_values):
				lookup[idx][val] = i if idx in self.attr_type_ids["categorical"] else val
		
		return lookup

	# convert categorical values to numbers
	def __label_to_number(self):
		dataset_size = len(self.dataset)
		row_size = len(self.dataset[0])
		for r in range(dataset_size):
			for c in range(row_size):
				self.dataset[r][c] = self.lookup[c][self.dataset[r][c]]

	# Split the dataset by class values, returns a dictionary
	def __separate_by_class(self, pos=-1):
		separated = dict()
		for i in range(len(self.dataset)):
			vector = self.dataset[i]
			class_value = vector[pos]
			if (class_value not in separated):
				separated[class_value] = list()
			separated[class_value].append(vector)
		return separated

	# Calculate the mean of a list of numbers
	def __mean(self, numbers):
		return sum(numbers)/float(len(numbers))

	# Calculate the standard deviation of a list of numbers
	def __stdev(self, numbers):
		avg = self.__mean(numbers)
		variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers) - 1)
		return sqrt(variance)

	# Calculate the mean, stdev and count for each column in a dataset
	def __summarize_dataset(self, rows):
		summaries = [(self.__mean(column), self.__stdev(column), len(column)) for column in zip(*rows)]
		del(summaries[-1])
		return summaries

	# Split dataset by class then calculate statistics for each row
	def __summarize_by_class(self, pos=-1):
		separated = self.__separate_by_class(pos)
		summaries = dict()
		for class_value, rows in separated.items():
			summaries[class_value] = self.__summarize_dataset(rows)
		return summaries

	# Calculate the Gaussian probability distribution function for x
	def __calculate_probability(self, x, mean, stdev):
		exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
		return (1 / (sqrt(2 * pi) * stdev)) * exponent

	# Calculate the probabilities of predicting each class for a given row
	def __calculate_class_probabilities(self, row):
		total_rows = sum([self.model[label][0][2] for label in self.model])
		probabilities = dict()
		for class_value, class_summaries in self.model.items():
			probabilities[class_value] = self.model[class_value][0][2]/float(total_rows)
			for i in range(len(class_summaries)):
				mean, stdev, size = class_summaries[i]

				if i in self.attr_type_ids["continuous"]:
					probabilities[class_value] *= self.__calculate_probability(row[i], mean, stdev)
				else: 
					probabilities[class_value] *= self.target_labels_counter[class_value][i][row[i]] / size

		return probabilities

	# Predict the class for a given row
	def predict(self, row):
		probabilities = self.__calculate_class_probabilities(row)
		best_label, best_prob = None, -1
		for class_value, probability in probabilities.items():
			if best_label is None or probability > best_prob:
				best_prob = probability
				best_label = class_value
		return best_label


	def process_input(self, row):
		for c in range(len(row)):
			if c in self.attr_type_ids["continuous"]:
				continue
			row[c] = self.lookup[c][row[c]]

	def get_labels_probabilities(self, row):
		likelihood_positive = self.__calculate_class_probabilities(row)[1]
		likelihood_negative = self.__calculate_class_probabilities(row)[0]
		
		positive = likelihood_positive / (likelihood_negative + likelihood_positive)
		negative = likelihood_negative / (likelihood_negative + likelihood_positive)

		return positive, negative

filename = os.path.join( os.path.dirname(__file__), "heart_disease_male.csv")
data = load_csv(filename)

bayes = NaiveBayes(dataset=data["dataset"],
					attr_type_ids=data["attr_type_ids"])

