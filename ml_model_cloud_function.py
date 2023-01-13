import pandas as pd
import numpy as np
import pickle
import json
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from sklearn.linear_model import LogisticRegression

def AUC_ROC(y_true, y_pred_proba, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    ref_line = np.linspace(0,1)

    plt.figure(figsize=(5,5))
    plt.title(f'ROC Curves - {model_name}')
    plt.plot(ref_line, ref_line, linestyle='dashed')
    plt.plot(fpr,tpr, label="AUC:"+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.show()

def distribution_plot(y_true, y_pred_proba, median, threshold, model_name):
    sns.displot(y_pred_proba)
    plt.title(f"Distribution Plot of Predicted Fall Risk, {model_name}")
    plt.axvline(median, color='red')
    plt.axvline(threshold, color='green')
    plt.plot();

def confusion(y_true, y_pred, model_name):
    conf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center')

    plt.xlabel('Predictions')
    plt.ylabel('Actuals')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

def evaluation(y_true, x, model, model_name): 
    y_pred_proba = model.predict_proba(x)[:,1]
    median = np.median(y_pred_proba)
    threshold = median*2
    
    AUC_ROC(y_test, y_pred_proba, model_name=model_name)
    distribution_plot(y_true, y_pred_proba, median=median, threshold=threshold, model_name=model_name)
    
    y_pred = (y_pred_proba >= threshold).astype(bool)
    confusion(y_test, y_pred, model_name=model_name)

def model_data(file):
    df = pd.read_csv(file)
    # Select Subset of columns
    df_clean = df[['ww_age', 'sex31', 'height50', 'own_rent680',
                 'ppl_in_household709', 'household_income738', 'grad_age845',
                 'alcohol_freq1558', 'health_rating2178', 'disability2188', 
                 'past_fall2296', 'operation2415', 'hearing_aid2293', 'visual_impair6148',
                 'employed6142', 'mild_depress_age20434', 'BMI21001', 'family_dementia20107',
                 'family_parki20107', 'family_severe_depression20107', 'months_btw_Fall']]

    df_clean = df_clean.copy()

    # Map own_rent to Own or not_own
    df_clean['own_rent680'] = df_clean["own_rent680"].map(lambda x: 1 if x == 1  else 0)

    # Median Imputation - ppl_in_household709, household_income738, grad_age845
    df_clean['ppl_in_household709'] = df_clean['ppl_in_household709'].fillna((df['ppl_in_household709'].median()))
    df_clean['household_income738'] = df_clean['household_income738'].fillna((df['household_income738'].median()))
    df_clean['grad_age845'] = df_clean['grad_age845'].fillna((df['grad_age845'].median()))

    # Drop Nan alcohol_freq1558
    df_clean = df_clean.dropna(subset=['alcohol_freq1558'])

    # Map disability2188 to yes=1 and no=0
    df_clean['disability2188'] = df_clean['disability2188'].map(lambda x: 0 if x == 1  else 1)

    # Map past_fall2296 to yes=1 and no=0
    df_clean['past_fall2296'] = df_clean['past_fall2296'].map(lambda x: 0 if x == 1  else 1)

    # Impute Nan to 0 for operation2415, hearing_aid2293, visual_impair6148
    df_clean['operation2415'] = df_clean['operation2415'].fillna(0)
    df_clean['hearing_aid2293'] = df_clean['hearing_aid2293'].fillna(0)
    df_clean['visual_impair6148'] = df_clean['visual_impair6148'].fillna(0)

    # Map cataract5441 to yes=1 and no=0, only 13ppl == 1
    # df_clean['cataract5441'] = df_clean['cataract5441'].fillna(0)
    # df_clean['cataract5441'] = df_clean['cataract5441'].map(lambda x: 1 if x != 0  else 0)

    # Drop Nan for glaucoma2137 as all are Nan
    # df_clean = df_clean.drop(columns=['glaucoma2137'])

    # Map employed6142 to 1=Employed, 2=Retired, 3=Housewife, 4=Unable to work, 5=Unemployed and others
    df_clean['employed6142'] = df_clean['employed6142'].fillna(5)
    df_clean['employed6142'] = df_clean['employed6142'].map(lambda x: 5 if x >= 5 else x)

    # Map mild_depress_age20434 to 0=No, 1=yes
    df_clean['mild_depress_age20434'] = df_clean['mild_depress_age20434'].fillna(0)
    df_clean['mild_depress_age20434'] = df_clean['mild_depress_age20434'].map(lambda x: 1 if x >= 1 else x)

    # Drop Nan, BMI21001
    df_clean = df_clean.dropna(subset=['BMI21001'])

    # Map Months Between Fall, if yes = 1, no = 0
    df_clean['months_btw_Fall'] = df_clean['months_btw_Fall'].fillna(0)
    df_clean['months_btw_Fall'] = df_clean['months_btw_Fall'].map(lambda x: 1 if x > 0 else 0).astype('int64')

    # Split num_col and cat_col
    num_col = ['ww_age', 'height50', 'grad_age845', 'alcohol_freq1558', 'health_rating2178', 
                 'employed6142', 'BMI21001']
    cat_col = set(list(df_clean.columns)) - set(num_col)

    num_col_idx = [df_clean.columns.get_loc(c) for c in df_clean.columns if c in num_col]
    cat_col_idx = [df_clean.columns.get_loc(c) for c in df_clean.columns if c in cat_col]

    X = df_clean.drop(columns=['months_btw_Fall'])
    y = df_clean['months_btw_Fall']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    # Linear Regression

    preprocessor = ColumnTransformer([
        ('pipe_num', Pipeline([('scaler', StandardScaler())]), num_col)],
        remainder='passthrough')

    lr_pipe = Pipeline([('preprocessor', preprocessor),
                     ('logistic_regression', LogisticRegression())])

    lr_pipe.fit(X_train, y_train)

    print("Median (%):", median*100)
    print("Prediction Threshold - 2 times risk (%):", threshold*100)
    print("Model Accuracy (%):", accuracy_score(y_test, y_pred)*100)
    print("Model Recall (%):", recall_score(y_test, y_pred)*100)

    evaluation(y_test, X_test, lr_pipe, "Linear Regression")

    coef_df = pd.DataFrame(data={'coefficient': lr_pipe['logistic_regression'].coef_.reshape(20,)},
                       index=df_clean.columns[:-1]).sort_values(by=['coefficient'], ascending=False)

    pickled_model = pickle.load(open('model.pkl', 'rb'))

    with open("test_instance.json", "r") as f:
    data = json.load(f)
    
    testing_df = pd.DataFrame([data])
    display(testing_df)

    test_pred = pickled_model.predict_proba(testing_df)[0][1]
    print(f"The test instance prediction result (%): {test_pred*100} ")



def handle_request(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <http://flask.pocoo.org/docs/1.0/api/#flask.Request>
    Returns:
        The result of the machine learning model.
    """
    # Get the file from the request
    file = request.files['file']

    # Read the file as a CSV file
    reader = csv.reader(file)
    input_data = []
    for row in reader:
        input_data.append(row)

    # Run the model and get the prediction
    prediction = model_data(input_data)

    return json.dumps({'prediction': prediction.numpy().tolist()})