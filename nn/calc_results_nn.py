import re 
import pandas as pd
import numpy as np

def find_duplicates(fbase, f2=None):
  fbase = set(line.strip() for line in open(fbase))
  fbase = sorted(fbase)
  for linebase in fbase:
    newLine = True
    count = 0
    for line in fbase:
      if line == linebase:
        count = count+1
    if count > 0:
      print(linebase)

def find_diff(f1, f2=None):
  f1 = set(line.strip() for line in open(f1))
  f2 = set(line.strip() for line in open(f2))
  f1 = sorted(f1)
  f2 = sorted(f2)
  count = 0
  for line1 in f1:
    # print(line1)
    foundOn2 = False
    for line2 in f2:
      if line1 == line2:
        foundOn2 = True
        break
    if foundOn2 == False:
      count = count +1
      print(f'Line not found {line1}')
  print(f'Lines not found {count}')

# find_duplicates('./full_test_set.txt')

# find_diff('./file1.txt','./file2.txt')

def calc_diff(df1, df2, compare):
  # res = pd.concat([df1, pd.concat([df2]*2)]).drop_duplicates(compare, keep=False)
  # res = df1[~df1.set_index(compare).index.isin(df2.set_index(compare).index)]
  # res =  (df1.reset_index() \
  #   .merge(df2, on=compare, indicator=True, how='outer', suffixes=('','_')) \
  #   .query('_merge == "left_only"') \
  #   .set_index('index').rename_axis(None).reindex(df1.columns, axis=1))
  res = df1[~df1[compare].astype(str).sum(axis=1).isin(df2[compare].astype(str).sum(axis=1))]
  return res

def transform_values_to_classification(df):
  conditions = [
      (df['R'] == 1.0),
      (df['L'] == 1.0),
      (df['B'] == 1.0)]
  choices = ['R', 'L', 'B']
  df['Class'] = np.select(conditions, choices)
  df.drop(['R', 'L', 'B'], axis='columns', inplace=True)
  return df

def process_data(df_data, df_results, _class):
  compareCols=["Class","Left-Weight","Left-Distance","Right-Weight","Right-Distance"]
  # Create Auxiliar dataset for TP FP TN FP comparisons
  conditions = [
      (df_data['Class'] == _class)
      ]
  choices = [_class]
  df_data['Class'] = np.select(conditions, choices, default=f'Not{_class}')
  #
  df_data_pos = df_data.loc[df_data['Class'] == _class]
  df_data_neg = df_data.loc[df_data['Class'] == f'Not{_class}']
  #
  df_results_pos = df_results.loc[df_results['Class'] == _class]
  df_results_neg = df_results.loc[df_results['Class'] == f'Not{_class}']
  #
  false_pos = calc_diff(df_results_pos, df_data_pos, compareCols)
  false_neg = calc_diff(df_results_neg, df_data_neg, compareCols)
  #
  true_pos = calc_diff(df_results_pos, false_pos, compareCols)
  true_neg = calc_diff(df_results_neg, false_neg, compareCols)
  #
  return false_pos, false_neg, true_pos, true_neg

def calc_prediction(df_data, df_results, _class, prediction_threshold=0.5):
  # Define R or NotR based on prediction and threshhold
  uniqueSet = ["Left-Weight","Left-Distance","Right-Weight","Right-Distance"]
  df_missing = calc_diff(df_data, df_results, uniqueSet)
  if not df_missing.empty:
    if prediction_threshold != 0.5: raise Exception('Faltam dados em pelo menos 1 dataset de resultados.')
    print('Calculating missing rows.')
    conditions = [
      (df_missing['Class'] == f'{_class}')
      ]
    choices = [f'Not{_class}']
    df_missing['Class'] = np.select(conditions, choices, default=f'{_class}')
    # Save dump
    df_missing.to_csv(f'./data/out/dump/{_path_code}/missing{_class}.csv', index=False)
  #
  conditions = [
      (df_results[f'PredictedValue{_class}'] >= 1.0 - prediction_threshold),
      (df_results[f'PredictedValue{_class}'] < prediction_threshold)
      ]
  choices = [_class, f'Not{_class}']
  df_results['Class'] = np.select(conditions, choices, default='-')
  unable = df_results.loc[df_results['Class'] == '-']
  df_results = df_results.loc[(df_results['Class'] != '-')]
  #
  if not df_missing.empty:
    print('Appending missing.')
    df_results = pd.concat([df_results, df_missing])
  #
  false_pos, false_neg, true_pos, true_neg = process_data(df_data, df_results, _class)
  #
  print(f'Class {_class} TP {len(true_pos)}; TN {len(true_neg)}; FP {len(false_pos)}; FN {len(false_neg)};')
  return len(false_pos), len(false_neg), len(true_pos), len(true_neg), len(unable)


# ========================================================================
# ====== MAIN ============================================================
# ========================================================================
SCALE_DATA_FILE = './data/scale.csv'
RESULT_DATA_PATH = './data/results/{}/result{}.csv'

prediction_threshold = 0.2 # 0.5 para cobrir todo o dataset
_path_codes = [
  # 'test_8hidden_0.1rate',
  'test_4hidden_0.1rate',
  # 'test_2hidden_0.1rate',
  # 'test_9hidden_0.1rate'
  ]

for _path_code in _path_codes:
  print(f'\n===================== {_path_code} =====================\n')
  for _class in ['R', 'L', 'B']:
    df_data = pd.read_csv(SCALE_DATA_FILE)
    df_results = pd.read_csv(RESULT_DATA_PATH.format(_path_code, _class))
    false_pos, false_neg, true_pos, true_neg, unable = calc_prediction(df_data, df_results, _class, prediction_threshold)
    print(f'Result over {len(df_data) - unable} instances, {unable} were unable to have a prediction with threshold of {prediction_threshold}.\n')
  print(f'===================== END =====================\n')