
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import boto3_rds_pandas as mysql
import joblib
import rds_ml_coefficients as rmc

print("runningk slearn_models.py")
def decision_tree(algo_object, X_train, X_test, y_train, y_test,objects_converted):
    print("running sklearn_models.decision_tree")
    objects_converted = str(objects_converted)
    objects_converted = objects_converted.replace("]", "").replace("[", "").replace("'", "")
    
    for key, value in algo_object.items():
        globals()[key] = value
    clf = DecisionTreeClassifier()
    # Train the classifier on the training data
    clf.fit(X_train, y_train)
    #y_col = y_train.columns
    #print(y_train)
    # Make predictions on the testing data
    y_pred = clf.predict(X_test)
    # Evaluate the classifier's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    #log algorithm
    query_insert = f"insert into stats.ml_algorithms (created_at,id,name,type,y_col,sql_used,object_fields_converted) values(now(),'{algo_id}','{algo_name}','{algo_type}','{y_col}','{query}','{objects_converted}')"
    mysql.run_sql_insert(query_insert)
    joblib.dump(clf, f'{algo_id}.joblib')
    
    return clf

        
def logistical(algo_object, X_train, X_test, y_train, y_test,objects_converted, df_x):
    print("running sklearn_models.logistical")
    for key, value in algo_object.items():
        globals()[key] = value
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    lr = LogisticRegression()
    # model = LogisticRegression(max_iter=100, solver='lbfgs') using different solvers and settting a max iterations
    lr_result = lr.fit(X_train,y_train)
    pred = lr.predict(X_test)
    #log algorithm
    #df_x = X_train

    objects_converted = str(objects_converted)
    objects_converted = objects_converted.replace("]", "").replace("[", "").replace("'", "")
    
    print('objects_converted',objects_converted)
    
    #print('****before failing insert')
    query_insert = f"insert into stats.ml_algorithms (created_at,id,name,type,y_col,sql_used,object_fields_converted) values(now(),'{algo_id}','{algo_name}','{algo_type}','{y_col}','{query}','{objects_converted}')"
    mysql.run_sql_insert(query_insert)
    joblib.dump(lr, f'{algo_id}.joblib')
    print('model created',lr)
    coefficients = lr.coef_
    intercept = lr.intercept_
    # Accessing other attributes
    penalty = lr.penalty
    solver = lr.solver
    C = lr.C
    # Inspecting the learned parameters
    parameters = lr.get_params()
    # Evaluate the model
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    parameters = json.dumps(parameters)
    precision = json.dumps(precision)
    query_update = f"update stats.ml_algorithms set parameters = '{parameters}', object = '{lr}', coefficients = '{coefficients}', intercept = '{intercept}',	penalty = '{penalty}',	solver = '{solver}',	C = '{C}',	accuracy = '{accuracy}', 	recall = '{recall}',	f1 = '{f1}',	roc_auc = '{roc_auc}'  where id ='{algo_id}' "
    mysql.run_sql_insert(query_update)
    #rmc.insert_ml_coefficents(df_x, coefficients)

    return lr
    
    
def elastic_net(algo_object, x_train, x_test, y_train, y_test,objects_converted):
    print("running sklearn_models.elastic_net")
    from sklearn.linear_model import ElasticNet
    from sklearn import metrics
    params = {
        "alpha": 0.3,    
        "fit_intercept": False,
        "l1_ratio": 0.7,
        "random_state": 0,
        "copy_X": False,
    }
    #start = timer()
    model = ElasticNet(**params).fit(x_train, y_train)
    #train_unpatched = timer() - start
    #f"Original Scikit-learn time: {train_unpatched:.2f} s"
    
    pred = model.predict(x_test)
    mse_metric_original = metrics.mean_squared_error(y_test, pred)
    f'Original Scikit-learn MSE: {mse_metric_original}'
    
    #unpack the algo_object
    for key, value in algo_object.items():
        globals()[key] = value
    
    #log algorithm
    query_insert = f"insert into stats.ml_algorithms (created_at,id,name,type,y_col,sql_used,object_fields_converted) values(now(),'{algo_id}','{algo_name}','{algo_type}','{y_col}','{query}','{objects_converted}')"
    mysql.run_sql_insert(query_insert)
    joblib.dump(model, f'{algo_id}.joblib')
    print('model created',model)
    coefficients = model.coef_
    intercept = model.intercept_
    # Accessing other attributes
    # penalty = model.penalty
    # solver = model.solver
    #C = model.C
    # Inspecting the learned parameters
    parameters = model.get_params()
    # Evaluate the model
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    
    #elasticnet specific params 
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, median_absolute_error, max_error
   
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    evs = explained_variance_score(y_test, pred)
    median_absolute_error = median_absolute_error(y_test, pred)
    max_error = max_error(y_test, pred)
    
    #roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    parameters = json.dumps(parameters)
    precision = json.dumps(precision)
    query_update = f"update stats.ml_algorithms set r_square = '{r2}', mse = '{mse}', mae = '{mae}', evs = '{evs}', median_absolute_error = '{median_absolute_error}',max_error = '{max_error}', parameters = '{parameters}',object = '{model}', coefficients = '{coefficients}', intercept = '{intercept}',	penalty = 'na',	solver = 'na',	C = 'na',	accuracy = '{accuracy}', 	recall = '{recall}',	f1 = '{f1}',	roc_auc = 'na'  where id ='{algo_id}' "
    mysql.run_sql_insert(query_update)

    return model 
    
    
#if __name__ == "__main__":
    #decision_tree()