from flask import  Flask,render_template,request
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import numpy as np                     # For mathematical calculations 



app = Flask(__name__)  


class MultiColumnLabelEncoder(LabelEncoder):
    """
    Wraps sklearn LabelEncoder functionality for use on multiple columns of a
    pandas dataframe.

    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, dframe):
        """
        Fit label encoder to pandas columns.

        Access individual column classes via indexig `self.all_classes_`

        Access individual column encoders via indexing
        `self.all_encoders_`
        """
        # if columns are provided, iterate through and get `classes_`
        if self.columns is not None:
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            self.all_encoders_ = np.ndarray(shape=self.columns.shape,
                                            dtype=object)
            for idx, column in enumerate(self.columns):
                # fit LabelEncoder to get `classes_` for the column
                le = LabelEncoder()
                le.fit(dframe.loc[:, column].values)
                # append the `classes_` to our ndarray container
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                # append this column's encoder
                self.all_encoders_[idx] = le
        else:
            # no columns specified; assume all are to be encoded
            self.columns = dframe.iloc[:, :].columns
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            for idx, column in enumerate(self.columns):
                le = LabelEncoder()
                le.fit(dframe.loc[:, column].values)
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                self.all_encoders_[idx] = le
        return self

   
    def transform(self, dframe):
        """
        Transform labels to normalized encoding.
        """
        if self.columns is not None:
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[
                    idx].transform(dframe.loc[:, column].values)
        else:
            self.columns = dframe.iloc[:, :].columns
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx]\
                    .transform(dframe.loc[:, column].values)
        return dframe.loc[:, self.columns].values

    def inverse_transform(self, dframe):
        """
        Transform labels back to original encoding.
        """
        if self.columns is not None:
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx]\
                    .inverse_transform(dframe.loc[:, column].values)
        else:
            self.columns = dframe.iloc[:, :].columns
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx]\
                    .inverse_transform(dframe.loc[:, column].values)
        return dframe.loc[:, self.columns].values

def pickle_file():
    loaded_model = pickle.load(open('label_encodings2', 'rb'))
    loaded_model1 = pickle.load(open('random_forest_model2', 'rb'))
    return loaded_model,loaded_model1
    
@app.route("/test", methods=["POST"]) 
def login():
    details = request.form
    print(details)
    return render_template('Input.html', title='Home')
    
@app.route('/')
def index():
    return render_template('Login.html', title='Home')


  

@app.route("/predict", methods=["POST"]) 
def predict():
    li = []
    details = request.form
    
    li.append(details['fname'])
    li.append(details['lname'])
    li.append(details['Sname'])
    li.append(details['Tname']) 
    li.append(details['Ename'])
    encoder, model = pickle_file()
    encodings = encoder.fit_transform(li)
    print(encodings)
    result = str(list(model.predict([encodings])))
    return render_template('Result.html', title='Home', result = result)
    #return str(list(model.predict([encodings]))[0]) #render_template('newww.html', title='Home')
    
@app.route("/out", methods=["POST"]) 
def login1():
    if request.method == 'POST':
        return result[0]

    
if __name__== '__main__':
    app.run(debug = True)
    
