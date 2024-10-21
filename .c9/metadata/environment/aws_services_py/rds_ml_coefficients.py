{"filter":false,"title":"rds_ml_coefficients.py","tooltip":"/aws_services_py/rds_ml_coefficients.py","undoManager":{"mark":48,"position":48,"stack":[[{"start":{"row":0,"column":0},"end":{"row":6,"column":78},"action":"insert","lines":["for col in num_cols:","    # figure out the stats needed to identify outliers","    min_v_clean = df[col].min()","    max_v_clean = df[col].max()","    mean_v_clean = df[col].mean()","    stdv_v_clean = df[col].std()","    min_max_data[col] = [min_v_clean, max_v_clean, mean_v_clean, stdv_v_clean]"],"id":1}],[{"start":{"row":0,"column":0},"end":{"row":1,"column":0},"action":"insert","lines":["",""],"id":2},{"start":{"row":1,"column":0},"end":{"row":2,"column":0},"action":"insert","lines":["",""]}],[{"start":{"row":0,"column":0},"end":{"row":5,"column":20},"action":"insert","lines":["import sys","# Surpress warnings:","def warn(*args, **kwargs):","    pass","import warnings","warnings.warn = warn"],"id":3}],[{"start":{"row":6,"column":0},"end":{"row":7,"column":0},"action":"insert","lines":["",""],"id":4},{"start":{"row":7,"column":0},"end":{"row":8,"column":0},"action":"insert","lines":["",""]},{"start":{"row":8,"column":0},"end":{"row":9,"column":0},"action":"insert","lines":["",""]},{"start":{"row":9,"column":0},"end":{"row":10,"column":0},"action":"insert","lines":["",""]},{"start":{"row":10,"column":0},"end":{"row":11,"column":0},"action":"insert","lines":["",""]}],[{"start":{"row":9,"column":0},"end":{"row":17,"column":26},"action":"insert","lines":["    df_x = df.drop(y_col, axis=1)","    predictors = df_x.columns","    ","    ","        # columns that are numerical","    num_cols = df.dtypes[df.dtypes != object]  # filtering by string categoricals","    num_cols = num_cols.index.tolist() ","    #filter y","    num_cols.remove(y_col)"],"id":5}],[{"start":{"row":19,"column":0},"end":{"row":19,"column":4},"action":"insert","lines":["    "],"id":6},{"start":{"row":20,"column":0},"end":{"row":20,"column":4},"action":"insert","lines":["    "]},{"start":{"row":21,"column":0},"end":{"row":21,"column":4},"action":"insert","lines":["    "]},{"start":{"row":22,"column":0},"end":{"row":22,"column":4},"action":"insert","lines":["    "]},{"start":{"row":23,"column":0},"end":{"row":23,"column":4},"action":"insert","lines":["    "]},{"start":{"row":24,"column":0},"end":{"row":24,"column":4},"action":"insert","lines":["    "]},{"start":{"row":25,"column":0},"end":{"row":25,"column":4},"action":"insert","lines":["    "]},{"start":{"row":26,"column":0},"end":{"row":26,"column":4},"action":"insert","lines":["    "]}],[{"start":{"row":8,"column":0},"end":{"row":8,"column":1},"action":"insert","lines":["d"],"id":7},{"start":{"row":8,"column":1},"end":{"row":8,"column":2},"action":"insert","lines":["e"]},{"start":{"row":8,"column":2},"end":{"row":8,"column":3},"action":"insert","lines":["f"]}],[{"start":{"row":8,"column":3},"end":{"row":8,"column":4},"action":"insert","lines":[" "],"id":8}],[{"start":{"row":8,"column":4},"end":{"row":8,"column":5},"action":"insert","lines":["i"],"id":9},{"start":{"row":8,"column":5},"end":{"row":8,"column":6},"action":"insert","lines":["n"]},{"start":{"row":8,"column":6},"end":{"row":8,"column":7},"action":"insert","lines":["s"]},{"start":{"row":8,"column":7},"end":{"row":8,"column":8},"action":"insert","lines":["e"]},{"start":{"row":8,"column":8},"end":{"row":8,"column":9},"action":"insert","lines":["r"]},{"start":{"row":8,"column":9},"end":{"row":8,"column":10},"action":"insert","lines":["t"]},{"start":{"row":8,"column":10},"end":{"row":8,"column":11},"action":"insert","lines":["_"]},{"start":{"row":8,"column":11},"end":{"row":8,"column":12},"action":"insert","lines":["m"]},{"start":{"row":8,"column":12},"end":{"row":8,"column":13},"action":"insert","lines":["l"]}],[{"start":{"row":8,"column":13},"end":{"row":8,"column":14},"action":"insert","lines":["_"],"id":10},{"start":{"row":8,"column":14},"end":{"row":8,"column":15},"action":"insert","lines":["c"]},{"start":{"row":8,"column":15},"end":{"row":8,"column":16},"action":"insert","lines":["o"]},{"start":{"row":8,"column":16},"end":{"row":8,"column":17},"action":"insert","lines":["e"]},{"start":{"row":8,"column":17},"end":{"row":8,"column":18},"action":"insert","lines":["f"]},{"start":{"row":8,"column":18},"end":{"row":8,"column":19},"action":"insert","lines":["f"]},{"start":{"row":8,"column":19},"end":{"row":8,"column":20},"action":"insert","lines":["i"]},{"start":{"row":8,"column":20},"end":{"row":8,"column":21},"action":"insert","lines":["c"]}],[{"start":{"row":8,"column":21},"end":{"row":8,"column":22},"action":"insert","lines":["e"],"id":11},{"start":{"row":8,"column":22},"end":{"row":8,"column":23},"action":"insert","lines":["n"]},{"start":{"row":8,"column":23},"end":{"row":8,"column":24},"action":"insert","lines":["t"]},{"start":{"row":8,"column":24},"end":{"row":8,"column":25},"action":"insert","lines":["s"]}],[{"start":{"row":8,"column":25},"end":{"row":8,"column":26},"action":"insert","lines":[" "],"id":12}],[{"start":{"row":8,"column":25},"end":{"row":8,"column":26},"action":"remove","lines":[" "],"id":13},{"start":{"row":8,"column":25},"end":{"row":8,"column":27},"action":"insert","lines":[". "]}],[{"start":{"row":8,"column":27},"end":{"row":8,"column":29},"action":"insert","lines":["()"],"id":14}],[{"start":{"row":8,"column":28},"end":{"row":8,"column":29},"action":"insert","lines":["d"],"id":15},{"start":{"row":8,"column":29},"end":{"row":8,"column":30},"action":"insert","lines":["f"]}],[{"start":{"row":8,"column":31},"end":{"row":8,"column":32},"action":"insert","lines":[":"],"id":16}],[{"start":{"row":8,"column":26},"end":{"row":8,"column":27},"action":"remove","lines":[" "],"id":17},{"start":{"row":8,"column":25},"end":{"row":8,"column":26},"action":"remove","lines":["."]}],[{"start":{"row":12,"column":0},"end":{"row":12,"column":4},"action":"remove","lines":["    "],"id":18},{"start":{"row":11,"column":4},"end":{"row":12,"column":0},"action":"remove","lines":["",""]}],[{"start":{"row":13,"column":17},"end":{"row":13,"column":18},"action":"insert","lines":["x"],"id":19}],[{"start":{"row":13,"column":28},"end":{"row":13,"column":29},"action":"insert","lines":["x"],"id":20}],[{"start":{"row":8,"column":28},"end":{"row":8,"column":29},"action":"insert","lines":["_"],"id":21},{"start":{"row":8,"column":29},"end":{"row":8,"column":30},"action":"insert","lines":["x"]}],[{"start":{"row":16,"column":3},"end":{"row":16,"column":26},"action":"remove","lines":[" num_cols.remove(y_col)"],"id":22}],[{"start":{"row":25,"column":82},"end":{"row":26,"column":0},"action":"insert","lines":["",""],"id":23},{"start":{"row":26,"column":0},"end":{"row":26,"column":8},"action":"insert","lines":["        "]},{"start":{"row":26,"column":8},"end":{"row":27,"column":0},"action":"insert","lines":["",""]},{"start":{"row":27,"column":0},"end":{"row":27,"column":8},"action":"insert","lines":["        "]}],[{"start":{"row":27,"column":8},"end":{"row":28,"column":45},"action":"insert","lines":["    query_insert4 = f\"insert into stats.ml_coefficients (id,column_name, coefficient, min, max, avg, stdv) values  ('{algo_id}','{predictor}','{coeff}','{v_min}','{v_max}','{v_avg}','{v_stdv}') \"","    df4 = mysql.run_sql_insert(query_insert4)"],"id":24}],[{"start":{"row":27,"column":8},"end":{"row":27,"column":12},"action":"remove","lines":["    "],"id":25}],[{"start":{"row":28,"column":4},"end":{"row":28,"column":8},"action":"insert","lines":["    "],"id":26}],[{"start":{"row":27,"column":20},"end":{"row":27,"column":21},"action":"remove","lines":["4"],"id":27}],[{"start":{"row":28,"column":10},"end":{"row":28,"column":11},"action":"remove","lines":["4"],"id":28}],[{"start":{"row":28,"column":46},"end":{"row":28,"column":47},"action":"remove","lines":["4"],"id":29}],[{"start":{"row":8,"column":30},"end":{"row":8,"column":31},"action":"insert","lines":[","],"id":30}],[{"start":{"row":8,"column":31},"end":{"row":8,"column":32},"action":"insert","lines":[" "],"id":31},{"start":{"row":8,"column":32},"end":{"row":8,"column":33},"action":"insert","lines":["p"]},{"start":{"row":8,"column":33},"end":{"row":8,"column":34},"action":"insert","lines":["r"]},{"start":{"row":8,"column":34},"end":{"row":8,"column":35},"action":"insert","lines":["e"]},{"start":{"row":8,"column":35},"end":{"row":8,"column":36},"action":"insert","lines":["d"]},{"start":{"row":8,"column":36},"end":{"row":8,"column":37},"action":"insert","lines":["i"]},{"start":{"row":8,"column":37},"end":{"row":8,"column":38},"action":"insert","lines":["c"]},{"start":{"row":8,"column":38},"end":{"row":8,"column":39},"action":"insert","lines":["t"]},{"start":{"row":8,"column":39},"end":{"row":8,"column":40},"action":"insert","lines":["o"]},{"start":{"row":8,"column":40},"end":{"row":8,"column":41},"action":"insert","lines":["r"]},{"start":{"row":8,"column":41},"end":{"row":8,"column":42},"action":"insert","lines":["s"]}],[{"start":{"row":8,"column":41},"end":{"row":8,"column":42},"action":"remove","lines":["s"],"id":32},{"start":{"row":8,"column":40},"end":{"row":8,"column":41},"action":"remove","lines":["r"]},{"start":{"row":8,"column":39},"end":{"row":8,"column":40},"action":"remove","lines":["o"]},{"start":{"row":8,"column":38},"end":{"row":8,"column":39},"action":"remove","lines":["t"]},{"start":{"row":8,"column":37},"end":{"row":8,"column":38},"action":"remove","lines":["c"]},{"start":{"row":8,"column":36},"end":{"row":8,"column":37},"action":"remove","lines":["i"]},{"start":{"row":8,"column":35},"end":{"row":8,"column":36},"action":"remove","lines":["d"]},{"start":{"row":8,"column":34},"end":{"row":8,"column":35},"action":"remove","lines":["e"]},{"start":{"row":8,"column":33},"end":{"row":8,"column":34},"action":"remove","lines":["r"]},{"start":{"row":8,"column":32},"end":{"row":8,"column":33},"action":"remove","lines":["p"]},{"start":{"row":8,"column":31},"end":{"row":8,"column":32},"action":"remove","lines":[" "]},{"start":{"row":8,"column":30},"end":{"row":8,"column":31},"action":"remove","lines":[","]}],[{"start":{"row":8,"column":30},"end":{"row":8,"column":31},"action":"insert","lines":[","],"id":33}],[{"start":{"row":8,"column":31},"end":{"row":8,"column":32},"action":"insert","lines":[" "],"id":34}],[{"start":{"row":8,"column":32},"end":{"row":8,"column":33},"action":"insert","lines":["c"],"id":35},{"start":{"row":8,"column":33},"end":{"row":8,"column":34},"action":"insert","lines":["o"]},{"start":{"row":8,"column":34},"end":{"row":8,"column":35},"action":"insert","lines":["e"]},{"start":{"row":8,"column":35},"end":{"row":8,"column":36},"action":"insert","lines":["f"]},{"start":{"row":8,"column":36},"end":{"row":8,"column":37},"action":"insert","lines":["f"]},{"start":{"row":8,"column":37},"end":{"row":8,"column":38},"action":"insert","lines":["i"]},{"start":{"row":8,"column":38},"end":{"row":8,"column":39},"action":"insert","lines":["c"]},{"start":{"row":8,"column":39},"end":{"row":8,"column":40},"action":"insert","lines":["i"]},{"start":{"row":8,"column":40},"end":{"row":8,"column":41},"action":"insert","lines":["e"]}],[{"start":{"row":8,"column":41},"end":{"row":8,"column":42},"action":"insert","lines":["n"],"id":36},{"start":{"row":8,"column":42},"end":{"row":8,"column":43},"action":"insert","lines":["t"]}],[{"start":{"row":8,"column":32},"end":{"row":8,"column":43},"action":"remove","lines":["coefficient"],"id":37},{"start":{"row":8,"column":32},"end":{"row":8,"column":37},"action":"insert","lines":["coeff"]}],[{"start":{"row":9,"column":4},"end":{"row":9,"column":33},"action":"remove","lines":["df_x = df.drop(y_col, axis=1)"],"id":38}],[{"start":{"row":13,"column":17},"end":{"row":13,"column":18},"action":"insert","lines":["_"],"id":39}],[{"start":{"row":13,"column":29},"end":{"row":13,"column":30},"action":"insert","lines":["_"],"id":40}],[{"start":{"row":21,"column":22},"end":{"row":21,"column":24},"action":"remove","lines":["df"],"id":41},{"start":{"row":21,"column":22},"end":{"row":21,"column":26},"action":"insert","lines":["df_x"]}],[{"start":{"row":22,"column":22},"end":{"row":22,"column":24},"action":"remove","lines":["df"],"id":42},{"start":{"row":22,"column":22},"end":{"row":22,"column":26},"action":"insert","lines":["df_x"]}],[{"start":{"row":23,"column":23},"end":{"row":23,"column":25},"action":"remove","lines":["df"],"id":43},{"start":{"row":23,"column":23},"end":{"row":23,"column":27},"action":"insert","lines":["df_x"]}],[{"start":{"row":24,"column":23},"end":{"row":24,"column":25},"action":"remove","lines":["df"],"id":44},{"start":{"row":24,"column":23},"end":{"row":24,"column":27},"action":"insert","lines":["df_x"]}],[{"start":{"row":9,"column":4},"end":{"row":9,"column":21},"action":"insert","lines":["min_max_data = {}"],"id":45}],[{"start":{"row":18,"column":0},"end":{"row":18,"column":4},"action":"remove","lines":["    "],"id":46},{"start":{"row":17,"column":0},"end":{"row":18,"column":0},"action":"remove","lines":["",""]},{"start":{"row":16,"column":3},"end":{"row":17,"column":0},"action":"remove","lines":["",""]},{"start":{"row":16,"column":2},"end":{"row":16,"column":3},"action":"remove","lines":[" "]}],[{"start":{"row":15,"column":0},"end":{"row":15,"column":13},"action":"remove","lines":["    #filter y"],"id":47},{"start":{"row":14,"column":39},"end":{"row":15,"column":0},"action":"remove","lines":["",""]}],[{"start":{"row":12,"column":0},"end":{"row":12,"column":36},"action":"remove","lines":["        # columns that are numerical"],"id":48},{"start":{"row":11,"column":4},"end":{"row":12,"column":0},"action":"remove","lines":["",""]}],[{"start":{"row":11,"column":0},"end":{"row":11,"column":4},"action":"remove","lines":["    "],"id":49},{"start":{"row":10,"column":29},"end":{"row":11,"column":0},"action":"remove","lines":["",""]}]]},"ace":{"folds":[],"scrolltop":0,"scrollleft":0,"selection":{"start":{"row":10,"column":29},"end":{"row":10,"column":29},"isBackwards":false},"options":{"guessTabSize":true,"useWrapMode":false,"wrapToView":true},"firstLineState":0},"timestamp":1715085025574,"hash":"85a6279fa7c1430ba9dadd8b8b042481dbc9ef5f"}