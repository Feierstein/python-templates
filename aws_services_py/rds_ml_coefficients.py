import sys
# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def insert_ml_coefficents(df_x, coeff):
    min_max_data = {}
    predictors = df_x.columns
    num_cols = df_x.dtypes[df_x.dtypes != object]  # filtering by string categoricals
    num_cols = num_cols.index.tolist() 
  
    for col in num_cols:
        # figure out the stats needed to identify outliers
        min_v_clean = df_x[col].min()
        max_v_clean = df_x[col].max()
        mean_v_clean = df_x[col].mean()
        stdv_v_clean = df_x[col].std()
        min_max_data[col] = [min_v_clean, max_v_clean, mean_v_clean, stdv_v_clean]
        
        query_insert = f"insert into stats.ml_coefficients (id,column_name, coefficient, min, max, avg, stdv) values  ('{algo_id}','{predictor}','{coeff}','{v_min}','{v_max}','{v_avg}','{v_stdv}') "
        df = mysql.run_sql_insert(query_insert)