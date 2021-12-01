import re 
import pandas as pd
import numpy as np

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

def calc_prediction(df_data, df_results, _class, max_sqr_err=1):
  conditions = [
    ((df_results[f'R'] > df_results[f'L']) & (df_results[f'R'] > df_results[f'B'])),
    ((df_results[f'L'] > df_results[f'R']) & (df_results[f'L'] > df_results[f'B'])),
    ((df_results[f'B'] > df_results[f'L']) & (df_results[f'B'] > df_results[f'R']))
    ]
  choices = ['R','L','B']
  df_results['ClassPredicted'] = np.select(conditions, choices, default='-')
  #
  unable = df_results.loc[df_results['ClassPredicted'] == '-']
  cut_by_err = df_results.loc[(df_results['S.Error**'] > max_sqr_err)]
  unused = df_results.loc[(df_results['ClassPredicted'] == '-') | (df_results['S.Error**'] > max_sqr_err)]
  df_results = df_results.loc[(df_results['ClassPredicted'] != '-') & (df_results['S.Error**'] <= max_sqr_err)]
  conditions = [
      (df_results['ClassPredicted'] == _class)
      ]
  choices = [_class]
  df_results['Class'] = np.select(conditions, choices, default=f'Not{_class}')
  #
  false_pos, false_neg, true_pos, true_neg = process_data(df_data, df_results, _class)
  #
  print(f'Class {_class} TP {len(true_pos)}; TN {len(true_neg)}; FP {len(false_pos)}; FN {len(false_neg)};')
  return len(false_pos), len(false_neg), len(true_pos), len(true_neg), len(unable), len(cut_by_err), len(unused)


# ========================================================================
# ====== MAIN ============================================================
# ========================================================================
SCALE_DATA_FILE = './data/scale.csv'
RESULT_DATA_PATH = './data/results/{}/result.csv'
TEST_DATA_PATH = './data/results/{}/test.csv'

max_sqr_err = 1 # erro 1 pra pegar todos os dados pra avaliar
_path_codes = [
  '016p030d',
  '03f234d'
  ]

print(f'\nTest dataset has {len(pd.read_csv(SCALE_DATA_FILE).index)} instances.')
for _path_code in _path_codes:
  for _class in ['R', 'L', 'B']:
    df_data = pd.read_csv(SCALE_DATA_FILE)
    df_results = pd.read_csv(RESULT_DATA_PATH.format(_path_code))
    false_pos, false_neg, true_pos, true_neg, unable, cut_by_err, unused = calc_prediction(df_data, df_results, _class, max_sqr_err)
  print(f'Result over {len(df_data) - unused} instances, {unable} were unable to have a prediction and {cut_by_err} had sqr error greater than {max_sqr_err}.\n')



# df_data = pd.read_csv(SCALE_DATA_FILE)
# df_results = pd.read_csv(RESULT_DATA_PATH.format(_path_codes[0]))


# joinCols=["Left-Weight","Left-Distance","Right-Weight","Right-Distance"]
# df_test = pd.read_csv(TEST_DATA_PATH.format(_path_codes[0]))

# test_df = pd.merge(df_test, df_results, on=joinCols)
