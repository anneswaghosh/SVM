import copy
import math
import numpy as np


def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

def data_in_x_y_format (filename, no_of_columns):
  no_of_rows = file_len (filename)
  data = np.zeros ((no_of_rows, no_of_columns))

  row_no = 0
  with open(filename) as f:
    for line in f:
      words = line.split()
      start = 1
      for word in words:
        if  start == 1:
          start = 0
          data[row_no][0] = int (word)
        else:
          parts = word.split(':')
          column = int (parts[0])
          value = float (parts[1])
          data[row_no][column] = value
      row_no += 1

  X = data
  Y = list(range (1, 220))
  return X, Y

class Node:
    def __init__(self, feature):
        self.feature = feature 
        self.children = []
        self.decision = ""

    def __str__(self):
        return self.feature

class Attribute:
	pass

class Data:
	
	def __init__(self, *, fpath = "", data = None):
		
		if not fpath and data is None:
			raise Exception("Must pass either a path to a data file or a numpy array object")

		self.raw_data, self.attributes, self.features, self.index_column_dict, \
		self.column_index_dict = self._load_data(fpath, data)

	def _load_data(self, fpath = "", data = None):
		
		if data is None:
			data = np.loadtxt(fpath, delimiter=',', dtype = str)

		header = data[0]
		index_column_dict = dict(enumerate(header))


		column_index_dict = {v: k for k, v in index_column_dict.items()}

		data = np.delete(data, 0, 0)

		attributes = self._set_attributes_info(index_column_dict, data)

		return data, attributes, header, index_column_dict, column_index_dict
	
	def _set_attributes_info(self, index_column_dict, data):
		attributes = dict()

		for index in index_column_dict:
			column_name = index_column_dict[index]
			if column_name == 'label':
				continue
			attribute = Attribute()
			attribute.name = column_name
			attribute.index = index - 1
			attribute.possible_vals = np.unique(data[:, index])
			attributes[column_name] = attribute

		return attributes

def information_gain_per_column (subset_data, features, col):
    dict = {}
    unique_labels = np.unique (subset_data[:, 0])
    if unique_labels.shape[0] == 1:
        label_entropy = 0
        return label_entropy
    else:
        count = np.zeros ((unique_labels.shape[0], 1))
        for i in range (0, unique_labels.shape[0]):                     # For each unique label
            for j in range (0, subset_data[:, 0].shape[0]):             # For each row in the label column
                if subset_data[j, 0] == unique_labels[i]:
                    count [i] += 1                                      # Count of +ve or -ve

    row_size = subset_data[:, 0].shape[0]

    i = 0
    label_entropy = 0
    for i in range (0, unique_labels.shape[0]):
        division_result = (count[i]/row_size)
        label_entropy -= (division_result) * math.log (division_result, 2)


    unique_values = np.unique (subset_data[:, col])


    i = 0
    no_of_occurence_per_value = np.zeros (unique_values.shape[0])

    for i in range (0, unique_values.shape[0]):
        for j in range (0, subset_data[:, 0].shape[0]):
            if subset_data [j, col] == unique_values [i]:
                no_of_occurence_per_value [i] += 1


    entropy_per_attr_value = np.zeros (unique_values.shape[0])
    for i in range (0, unique_values.shape[0]):
        count = np.zeros ((unique_labels.shape[0], 1))
        for j in range (0, unique_labels.shape[0]):
            for k in range (0, subset_data[:, 0].shape[0]):
                if subset_data [k, col] == unique_values[i]:
                    if subset_data [k, 0] == unique_labels[j]:
                        count [j] += 1
        for j in range (0, unique_labels.shape[0]):
            if count[j] != 0:
                division_result = (count[j]/no_of_occurence_per_value[i])
                entropy_per_attr_value [i] -= (division_result) * math.log (division_result, 2)

    expected_entropy_for_column = 0
    for i in range (0, unique_values.shape[0]):
        expected_entropy_for_column += (no_of_occurence_per_value[i]/row_size) * entropy_per_attr_value [i]


    info_gain = label_entropy - expected_entropy_for_column

    return info_gain 

def information_gain_all_columns (subset_data, features):
    no_of_col = subset_data.shape[1]

    info_gain_per_column = np.zeros (no_of_col)

    for col in range (1, no_of_col):
        info_gain_per_column[col] = information_gain_per_column (subset_data, features, col)

    return info_gain_per_column 

def get_next_feature (subset_data, features):
    info_gain_per_column = information_gain_all_columns (subset_data, features)
    next_col_index = np.argmax (info_gain_per_column)
    return next_col_index, features[next_col_index] 

def get_row_subset(data, col_index, attribute_value):
    subset_data = copy.deepcopy(data)
    subset_data = subset_data [subset_data[:, col_index] == attribute_value]
    return subset_data

def add_node (subset_data, features, current_depth, allowed_depth):
    if current_depth >= allowed_depth:
        unique_labels = np.unique (subset_data[:, 0])
        if unique_labels.shape[0] == 1:
            node = Node ("")
            node.decision = np.unique (subset_data[:, 0])[0]
            return node
        else:
            count = np.zeros ((unique_labels.shape[0], 1))
            for i in range (0, unique_labels.shape[0]):                     # For each unique label
                for j in range (0, subset_data[:, 0].shape[0]):             # For each row in the label column
                    if subset_data[j, 0] == unique_labels[i]:
                        count [i] += 1                                      # Count of +ve or -ve
        decision_index = np.argmax (count)
        node = Node ("")
        node.decision = unique_labels[decision_index]
        return node
 
    unique_labels = np.unique (subset_data[:, 0]).shape[0]
    if unique_labels == 1:
        node = Node ("")
        node.decision = np.unique (subset_data[:, 0])[0]
        return node
    selected_col_index, selected_feature = get_next_feature (subset_data, features)
    selected_feature_unique_values = np.unique (subset_data[:, selected_col_index])

    node = Node (selected_feature)
    for i in range (0, selected_feature_unique_values.shape[0]):
        new_subset = get_row_subset (subset_data, selected_col_index, selected_feature_unique_values[i]) 
        new_subset = np.delete(new_subset, selected_col_index, 1)
        new_feature_subset = copy.deepcopy (features)
        new_feature_subset = np.delete (new_feature_subset, selected_col_index, 0) 
        child_node = add_node (new_subset, new_feature_subset, current_depth+1, allowed_depth)
        node.children.append((selected_feature_unique_values[i], child_node))

    return node

def test_dtree_per_row (node, test_row, col_index_feature_dict, feature_col_index_dict):
    if node.decision != "":
        return node.decision

    col_index = feature_col_index_dict[node.feature]
    col_value = test_row[col_index]

    for child in node.children :
        if col_value == child[0]:
            return test_dtree_per_row (child[1], test_row, col_index_feature_dict, feature_col_index_dict)

def test_dtree(root, test_data, features):
    no_of_rows = test_data.shape[0]
    index_column_dict = dict(enumerate(features))
    column_index_dict = {v: k for k, v in index_column_dict.items()}
    mismatch = 0
    match = 0
    for i in range (0, no_of_rows):
        id3_result = test_dtree_per_row (root, test_data [i], index_column_dict, column_index_dict)
        if id3_result != test_data [i][0]:
            mismatch += 1
        else:
            match += 1
    
    accuracy = match/no_of_rows
    return (accuracy*100)
