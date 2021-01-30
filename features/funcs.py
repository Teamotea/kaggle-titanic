import pandas as pd

def load_feather(df1, file_name, col_name):
  df2 = pd.read_feather(file_name)
  df2 = df2[['PassengerId', col_name]]
  if col_name in df1.columns:
    df1 = df1.drop(col_name, axis=1)
  df = pd.merge(df1, df2, on='PassengerId', how='inner')
  return df