import copy
import math
import numpy as np
import decision_tree as dtree

def lr_test(X, Y, W):
  rows = X.shape[0]
  true_pos = 0
  false_pos = 0
  false_neg = 0
  errors = 0

  for i in range (0, rows):
    val = (np.dot(X[i], W))
    if (val * Y[i,0]) < 0:
      errors += 1
      if (Y[i,0] > 0):
        false_neg += 1
      else:
        false_pos += 1
    else:
      if (Y[i,0] > 0):
        true_pos += 1

  if (errors == 0):
    true_pos = 0
    false_pos = 0
    false_neg = 0
  return true_pos, false_pos, false_neg

def train_lr(X, Y, W, tradeoff, l_rate, epochs, count_corrections):
  update_count = 0
  cols = X.shape[1]
  rows = X.shape[0]
  for t in range(0, epochs):
    randomize = np.arange (X.shape[0])
    np.random.shuffle(randomize)
    X = X[randomize]
    Y = Y[randomize]
    rate = (l_rate/(1+t))

    for i in range (0, rows):
      try:
        gradient = ((-1 * X[i] * Y[i,0])/(1 + math.exp (Y[i] * np.dot (W, X[i])))) + ((2*W)/(tradeoff))
      except OverflowError:
        gradient = ((2*W) / tradeoff)

      W = W - rate * gradient
  return W

def lr_train_test (train_filename, test_filename, no_of_columns, W, C,
                        l_rate, epochs, count_corrections):

  if ("transformed_" in test_filename):
    transformed = True
  else:
    transformed = False

  if (transformed == True):
    X, Y = transformed_get_data_in_x_y (train_filename, no_of_columns)
    new_W = train_lr(X, Y, W, C, l_rate, epochs, count_corrections)

    X, Y = transformed_get_data_in_x_y (test_filename, no_of_columns)
    true_pos, false_pos, false_neg = lr_test (X, Y, new_W)
  else:
    X, Y = get_data_in_x_y (train_filename, no_of_columns)
    new_W = train_lr(X, Y, W, C, l_rate, epochs, count_corrections)

    X, Y = get_data_in_x_y (test_filename, no_of_columns)
    true_pos, false_pos, false_neg = lr_test (X, Y, new_W)

  print (" True Positive  :", true_pos, " False Positive :", false_pos, " False Negative :", false_neg)

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
  print (" F Value       ", F1)
  print (" Precision     ", precision)
  print (" Recall        ", recall)
  print ("")
  return new_W, precision, recall, F1 

def get_f_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

def transformed_get_data_in_x_y (filename, no_of_columns):
  data = np.load (filename)
  raw_Y = copy.deepcopy(data)
  Y = np.delete (raw_Y, np.s_[1:no_of_columns], axis=1)
  X = np.delete (data, 0, axis=1)
  return X, Y

def get_data_in_x_y (filename, no_of_columns):
  no_of_rows = get_f_len (filename)
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

def cross_validation (kfold, C, l_rate, epochs, no_of_columns, W, partial_filename):
  precision = 0
  consolidated_F1 = 0

  if ("transformed_" in partial_filename):
    transformed = True
  else:
    transformed = False

  for i in range (0, kfold):
    training_filenames = []
    temp_arr_start = True
    for j in range (0, kfold):
      if (i != j):
        training_filenames.append (partial_filename + str(j)+'.data')

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

    new_W, precision, recall, F1 = lr_train_test ('temporary.data', partial_filename+str(i)+'.data',
                                                  no_of_columns, W, C, l_rate, epochs, 0)
    consolidated_F1 += F1 
  return (consolidated_F1/kfold)

def lr_test_begin (kfold, learn_rates, tradeoff_params, epochs,
                                  no_of_columns, W):
  selected_f1 = 0
  for C in tradeoff_params:
    for l_rate  in learn_rates:
      print ("")
      print (" Cross-validation running for C ", C, "Rate : ", l_rate)
      print ("----------------------------------------------------------------")
      W_copy = copy.deepcopy (W)
      f1 =  cross_validation (kfold, C, l_rate, epochs, no_of_columns, W_copy, "training0")
      if (f1 > selected_f1):
        selected_f1 = f1 
        selected_tradeoff = C
        selected_l_rate = l_rate 


  print ("----------------------------------------------------------------")
  print ("Cross validation output")
  print ("   Selected Learning Rate   ", selected_l_rate)
  print ("   Selected tradeof Param   ", selected_tradeoff)
  print ("   Yielded F1           ", selected_f1)
  print ("----------------------------------------------------------------")
  selected_f1 = 0
  selected_epoch = 0
  selected_precision = 0
  selected_recall = 0
  selected_w = np.zeros(no_of_columns - 1)
  
  print ("Test begins")
  for i in range (1, 20):
    print ("")
    print (" Epoch      ", i)
    new_W, precision, recall, f1 = lr_train_test ('train.liblinear', 'test.liblinear', no_of_columns,
                                                   W, selected_tradeoff, selected_l_rate, i, 0)
    print (" Precision  :", precision , " Recall :", recall, " F1 :", f1)
    if (f1 > selected_f1):
      selected_f1 = f1 
      selected_precision = precision
      selected_recall = recall
      selected_epoch = i
      selected_w = copy.deepcopy (new_W)

  print ("----------------------------------------------------------------")
  print ("   Selected epoch                    ", selected_epoch)
  print ("   Selected F1                       ", selected_f1)
  print ("----------------------------------------------------------------")

  return selected_f1 

seed_value = 100
kfold           = 5
no_of_columns   = 220
np.random.seed (seed_value)
W               = np.zeros (no_of_columns-1)
epochs          = 10 
precision       = 0
learn_rates     = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
tradeoff_params = [0.1, 1, 10, 100, 1000, 10000]
lr_test_begin (kfold, learn_rates, tradeoff_params, epochs, no_of_columns, W)
