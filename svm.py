import copy
import math
import numpy as np
import decision_tree as dtree

def test_svm(X, Y, W):
  rows = X.shape[0]
  mistakes = 0
  false_neg = 0
  false_pos = 0
  true_pos = 0

  for i in range (0, rows):
    val = (np.dot(X[i], W))
    if (val * Y[i,0]) < 0:
      mistakes += 1
      if (Y[i,0] > 0):
        false_neg += 1
      else:
        false_pos += 1
    else:
      if (Y[i,0] > 0):
        true_pos += 1

  return true_pos, false_pos, false_neg

def svm_learner(X, Y, W, C, learning_rate, no_of_epochs, count_mistakes):
    update_count = 0
    cols = X.shape[1]
    rows = X.shape[0]
    for t in range(0, no_of_epochs):
        randomize = np.arange (X.shape[0])
        np.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]
        rate = (learning_rate/(1+t))
        
        for i in range (0, rows):
          if ((np.dot(X[i], W)*Y[i,0]) <= 1):
            W = ((1-rate)*W) + (rate * C * Y[i,0] * X[i])
          else:
            W = (1-rate)*W

    return W

def svm_training_and_testing (training_filename, testing_filename, no_of_columns, W, C,
                        learning_rate, no_of_epochs, count_mistakes):
  if ("transformed_" in testing_filename):
    transformed = True
  else:
    transformed = False

  if (transformed == True):
    X, Y = get_tranformed_data_in_x_y (training_filename, no_of_columns)
    new_W = svm_learner(X, Y, W, C, learning_rate, no_of_epochs, count_mistakes)

    X, Y = get_tranformed_data_in_x_y (testing_filename, no_of_columns)
    true_pos, false_pos, false_neg = test_svm (X, Y, new_W)
  else:
    X, Y = get_data_in_x_y (training_filename, no_of_columns)
    new_W = svm_learner(X, Y, W, C, learning_rate, no_of_epochs, count_mistakes)

    X, Y = get_data_in_x_y (testing_filename, no_of_columns)
    true_pos, false_pos, false_neg = test_svm (X, Y, new_W)

  print (" True Positive   : ", true_pos)
  print (" False Positive  : ", false_pos)
  print (" False Negative  : ", false_neg)

  if (true_pos != 0) or (false_pos != 0):
    precision = (true_pos/(true_pos+false_pos))
  else:
    precision = 0

  if (true_pos != 0) or (false_neg != 0):
    recall = (true_pos/(true_pos+false_neg))
  else:
    recall = 0

  if (precision != 0) or (recall != 0):
    F1 = 2 * ((precision * recall) / (precision + recall))
  else:
    F1 = 0
  print (" F Value    : ", F1)
  print (" Precision  : ", precision)
  print (" Recall     : ", recall)
  print ("")
  return new_W, precision, recall, F1 

def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

def retrieve_d_and_f (filename, no_of_columns):
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
  Y = list (range (0,220))
  return X, Y

def get_tranformed_data_in_x_y (filename, no_of_columns):
  data = np.load (filename)
  raw_Y = copy.deepcopy(data)
  Y = np.delete (raw_Y, np.s_[1:no_of_columns], axis=1)
  X = np.delete (data, 0, axis=1)
  return X, Y

def get_data_in_x_y (filename, no_of_columns):
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

  raw_Y = copy.deepcopy(data)
  Y = np.delete (raw_Y, np.s_[1:no_of_columns], axis=1)
  X = np.delete (data, 0, axis=1)
  return X, Y

def cross_validation (no_of_folds, C, learning_rate, no_of_epochs, no_of_columns, W, fname_partial):
  precision = 0
  consolidated_F1 = 0

  if ("transformed_" in fname_partial):
    transformed = True
  else:
    transformed = False

  for i in range (0, no_of_folds):
    training_filenames = []
    temp_arr_start = True
    for j in range (0, no_of_folds):
      if (i != j):
        training_filenames.append (fname_partial + str(j)+'.data')

    if (transformed == False):
      with open ('temporary.data', 'w') as temp_file:
        for fname  in training_filenames:
          with open(fname) as iterfile:
            for line in iterfile:
              temp_file.write (line)
    else:
      for fname in training_filenames:
        transient_arr = np.load (fname)
        if temp_arr_start == True:
          temp_arr_start = False
          temp_arr = copy.deepcopy (transient_arr)
        else:
          temp_arr = np.concatenate ((temp_arr, transient_arr))
      temp_arr.dump ("temporary.data")

    #Cross Validation Training
    new_W, precision, recall, F1 = svm_training_and_testing ('temporary.data', fname_partial+str(i)+'.data',
                                                  no_of_columns, W, C, learning_rate, no_of_epochs, 0)
    consolidated_F1 += F1 
  return (consolidated_F1/no_of_folds)

def initiate_testing (no_of_folds, learning_rates, C_values, no_of_epochs,
                                  no_of_columns, W):
  selected_f1 = 0

  for C in C_values:
    for learning_rate  in learning_rates:
      print ("")
      print (" Running Cross validation for Tradeoff : ", C, "Learning Rate : ", learning_rate)
      W_copy = copy.deepcopy (W)
      f1 =  cross_validation (no_of_folds, C, learning_rate, no_of_epochs, no_of_columns, W_copy, "training0")
      if (f1 > selected_f1):
        selected_f1 = f1 
        selected_C = C
        selected_learning_rate = learning_rate 


  print ("-----------------------------------------------")
  print ("Cross validation results ")
  print ("   Selected Learning Rate  : ", selected_learning_rate)
  print ("   Selected tradeof Param  : ", selected_C)
  print ("   Yielded F1          : ", selected_f1)
  print ("-----------------------------------------------")
  # Re-init for future use
  selected_f1 = 0
  selected_epoch = 0
  selected_precision = 0
  selected_recall = 0
  selected_w = np.zeros(no_of_columns - 1)

  print ("Test begins")
  #Train for each epoch and test in development data for each of them and measure accuracy
  for i in range (1, 21):
    print ("")
    print (" Epoch      :", i)
    new_W, precision, recall, f1 = svm_training_and_testing ('train.liblinear', 'test.liblinear', no_of_columns,
                                                   W, selected_C, best_learning_rate, i, 0)
    print (" Precision  : ", precision)
    print (" Recall     : ", recall)
    print (" F1         : ", f1)
    if (f1 > selected_f1):
      selected_f1 = f1 
      selected_precision = precision
      selected_recall = recall
      selected_epoch = i
      selected_w = copy.deepcopy (new_W)

  print ("-----------------------------------------------#")
  print ("   Selected Epoch                   : ", selected_epoch)
  print ("   Selected F1                      : ", selected_f1)
  print ("-----------------------------------------------#")

  return selected_f1 

def random_forest (training_file, sample_draw_size, no_of_dtree, no_of_columns, depth):
  X, features = retrieve_d_and_f (training_file, no_of_columns)
  randomize = np.arange (X.shape[0])

  print ("Dtree Depth : ", depth)
  dtree_list = []
  for i in range (0, no_of_dtree):
    np.random.shuffle(randomize)
    X = X[randomize]
    sample = X[0:sample_draw_size]
    print ("Generating Dtree", i)
    root = dtree.add_node (sample, features, 0, depth)
    dtree_list.append(root)

  return dtree_list

def transform_features_with_random_forest (input_file, transformed_file, no_of_columns_in_input, dtree_list): 
  X, features = retrieve_d_and_f (input_file, no_of_columns_in_input)
  no_of_rows = X.shape[0]
 
  no_of_dtree = len (dtree_list)
  transformed_data = np.zeros ((no_of_rows, no_of_dtree+1))

  index_column_dict = dict(enumerate(features))
  column_index_dict = {v: k for k, v in index_column_dict.items()}
  for i in range (0, no_of_rows):
    print ("Transforming train row : ", i)
    transformed_data[i][0] = X[i][0]
    for j in range (0, no_of_dtree):
      root = dtree_list[j]
      transformed_data[i][1+j] = dtree.test_dtree_per_row (root, X[i], index_column_dict, column_index_dict)

  transformed_data.dump (transformed_file)

def svm_with_dtrees (training_file, testing_file, sample_draw_size, dtrees_hyp, depths, no_of_folds, learning_rates, C_values, no_of_epochs, no_of_columns, W):
  selected_learning_rate = 0
  selected_C      = 0
  selected_depth  = 0
  selected_f1     = 0
  W           = np.zeros (dtrees_hyp)

  for depth in depths:
    dtree_list = random_forest(training_file, sample_draw_size, dtrees_hyp, no_of_columns, depth)
    
    transform_features_with_random_forest ("training00.data", "transformed_training00.data", no_of_columns, dtree_list) 
    transform_features_with_random_forest ("training01.data", "transformed_training01.data", no_of_columns, dtree_list) 
    transform_features_with_random_forest ("training02.data", "transformed_training02.data", no_of_columns, dtree_list) 
    transform_features_with_random_forest ("training03.data", "transformed_training03.data", no_of_columns, dtree_list) 
    transform_features_with_random_forest ("training04.data", "transformed_training04.data", no_of_columns, dtree_list) 
    for C in C_values:
      for learning_rate  in learning_rates:
        print ("")
        print (" Running Cross validation for Depth : ", depth, "C: ", C, "Rate : ", learning_rate)
        print ("------------------------------------------------------")
        W_copy = copy.deepcopy (W)
        f1 =  cross_validation (no_of_folds, C, learning_rate, no_of_epochs, dtrees_hyp+1, W_copy, "transformed_training0")
      
        if (f1 > selected_f1):
          selected_f1 = f1 
          selected_C = C
          selected_learning_rate = learning_rate 
          selected_depth = depth

  print ("-----------------------------------------------")
  print ("Cross validation output")
  print ("   Depth          : ", selected_depth)
  print ("   Learning Rate  : ", selected_learning_rate)
  print ("   Tradeof Param  : ", selected_C)
  print ("   F1             : ", selected_f1)
  print ("-----------------------------------------------")

  transformed_training_file = "transformed_train.liblinear"
  transformed_testing_file  = "transformed_test.liblinear"
  transform_features_with_random_forest (training_file, transformed_training_file, no_of_columns, dtree_list) 
  transform_features_with_random_forest (testing_file, transformed_testing_file, no_of_columns, dtree_list) 
  print (" SVM over Trees test results")
  selected_f1 = 0
  for i in range (1, 21):
    print ("")
    print (" Epoch      :", i)
    new_W, precision, recall, f1 = svm_training_and_testing (transformed_training_file, transformed_testing_file, no_of_columns,
                                                       W, selected_C, best_learning_rate, i, 0)
    print (" Precision  : ", precision)
    print (" Recall     : ", recall)
    print (" F1         : ", f1)
    if (f1 > selected_f1):
      selected_f1 = f1 
      selected_precision = precision
      selected_recall = recall
      selected_epoch = i
      selected_w = copy.deepcopy (new_W)

  print ("-----------------------------------------------#")
  print ("   Selected epoch              : ", selected_epoch)
  print ("   Selected F1                 : ", selected_f1)
  print ("-----------------------------------------------#")

  return selected_f1 

seed_value = 100
learning_rates     = [1, 0.1, 0.01, 0.001, 0.0001]
C_values = [10, 1, 0.1, 0.01, 0.001, 0.0001]
depths          = [10, 20, 30]
training_file      = "train.liblinear"
testing_file       = "test.liblinear"
no_of_columns   = 220
no_of_folds           = 5
np.random.seed (seed_value)
precision       = 0
sample_draw_size     = 2000
dtrees_hyp    = 200
W               = np.zeros (no_of_columns-1)
no_of_epochs          = 10 

precision = initiate_testing (no_of_folds, learning_rates, C_values,
                              no_of_epochs, no_of_columns, W)
svm_with_dtrees (training_file, testing_file, sample_draw_size, dtrees_hyp,
                 depths, no_of_folds, learning_rates, C_values, no_of_epochs,
                 no_of_columns, W)


