from utilities.std_imports import *
from datetime import datetime
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


# Show columns of dataframe as time series charts 
def show_tss(df):
    cols = range(len(df.columns))
    i = 1
    plt.figure(figsize=(20,10))
    for c in cols:
        plt.subplot(len(cols), 1, i)
        plt.plot(df.values[:, c])
        plt.title(df.columns[c], x=0.005, y=0.8, loc='left')
        i += 1
    plt.show()

    
# Generate variable names for dataframe column headers  (col : variable (column) name, t : +-time int )  
def var_name(col, t):
    vn = col + '(t'
    if t>0: 
        vn = vn + '+'+ str(t)
    elif t<0: 
        vn = vn + str(t) 
    return vn + ')'


# Build dataset to train uni/multivariate models (vobs: observable column names, vreg: regressor column names, fh: forecast horizon )
def build_ds(df, vobs=[], vreg=[], mw=1, fh=1, dropnan=True):
    
    vars = vobs.copy()
    vars.extend(vreg)
    
    colnames, columns = list(), list()
    
    # hist (t-n, ... t-1)
    for i in range(mw, 0, -1):
        colnames += [(var_name(v, -i)) for v in vars]
        columns.append(df[vars].shift(i, axis=0))
    
    # fcst (t, ... t+n)
    for i in range(0, fh):
        colnames += [(var_name(v, +i)) for v in vobs]
        columns.append(df[vobs].shift(-i, axis=0))
        
    ds = pd.concat(columns, axis=1)
    ds.columns = colnames
    
    if dropnan:
        ds.dropna(inplace=True)
    return ds


# Prepare for training: Split train/test, Xy, reshape (split_index: index to split horizontally train/test, steps: timesteps for reshaping every variable)
def prep_train(ds, split_index, nvars, nsteps):
    nlabels = nvars*nsteps
    ds_train = ds.values[:split_index, :]
    ds_test = ds.values[split_index:, :]

    X_train, y_train = ds_train[:, :-nlabels], ds_train[:, -nlabels:]
    X_test, y_test = ds_test[:, :-nlabels], ds_test[:, -nlabels:]

    return X_train, y_train, X_test, y_test 


# Reshape for LSTM 3D format [nsamples, nsteps, nvars]
def reshape4lstm(X_train, X_test, mw):
     nvars = (int)(X_train.shape[1]/mw)
     return X_train.reshape(X_train.shape[0], mw, nvars), X_test.reshape(X_test.shape[0], mw, nvars)
    
