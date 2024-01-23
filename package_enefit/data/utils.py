import numpy as np
import pandas as pd


## changing a 2D dataframe into a 3D array with (n prediction units, n time steps,n features) shape

def df_to_ndarray(df,dim1,dim2) :
    """
    change a df to 3D array by using dim1 and dim2 as the 2 first dimensions,
    the 3rd dimension is composed of all the other features of the dataframe
    """
    iix_n = pd.MultiIndex.from_product([np.unique(df.dim1), np.unique(df.dim2)])
    array = (df.pivot_table(list(df.columns).remove([dim1,dim2]),[dim1,dim2])
         .reindex(iix_n).to_numpy()
         .reshape(df[dim1].nunique(), df[dim2].nunique(),-1))
    return array
