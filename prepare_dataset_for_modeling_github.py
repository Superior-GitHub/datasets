# V. Aksakalli
# GPL-3.0, 2020

import pandas as pd
import os
import io
import requests
import ssl
from sklearn import preprocessing
from sklearn.utils import shuffle


def prepare_dataset_for_modeling(dataset_name,
                                 data_directory=None,
                                 n_samples=None,
                                 random_state=999,
                                 drop_const_columns=True,
                                 scale_x=True,
                                 is_classification=True):
    """
    :param dataset_name: name of the dataset (in CSV format)
    :param data_directory: directory of the dataset. If None, the dataset will be read in from GitHub
    :param n_samples: how many instances to sample (if not None)
    :param random_state: seed for (1) sampling instances and (2) shuffling instances at the end
    :param drop_const_columns: if True, drop constant-value columns (*after* any sampling)
    :param scale_x: whether the descriptive features are to be min-max scaled
    :param is_classification: if True, y values will be label-encoded for use in classification models
    :return: x and y NumPy arrays ready for model fitting
    """

    if data_directory:
        # read in from local directory
        df = pd.read_csv(data_directory + dataset_name, header=0)
    else:
        # read in the data file from GitHub into a Pandas data frame
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
                getattr(ssl, '_create_unverified_context', None)):
            ssl._create_default_https_context = ssl._create_unverified_context
        github_location = 'https://raw.githubusercontent.com/vaksakalli/datasets/master/'
        dataset_url = github_location + dataset_name.lower()
        df = pd.read_csv(io.StringIO(requests.get(dataset_url).content.decode('utf-8')), header=0)

    # drop missing values if there are any
    df = df.dropna()

    # shuffle dataset in case of a pattern and subsample if requested
    df = shuffle(df, n_samples=n_samples, random_state=random_state)

    if drop_const_columns:
        # drop constant columns
        df = df.loc[:, df.nunique() > 1]

    # last column is y (target feature)
    y = df.iloc[:, -1]
    # everything else is x (set of descriptive features)
    x = df.iloc[:, :-1]

    # get all columns that are strings
    # these are assumed to be nominal categorical
    categorical_cols = x.columns[x.dtypes == object].tolist()

    # if a nominal feature has only 2 levels:
    # encode it as a single binary variable
    for col in categorical_cols:
        n = len(x[col].unique())
        if n == 2:
            x[col] = pd.get_dummies(x[col], drop_first=True)

    # for categorical features with >2 levels: use one-hot-encoding
    # below, numerical columns will be untouched
    x = pd.get_dummies(x).values

    if scale_x:
        # scale x between 0 and 1
        x = preprocessing.MinMaxScaler().fit_transform(x)

    if is_classification:
        # label-encode y for classification problems
        y = preprocessing.LabelEncoder().fit_transform(y)

    return x, y

# ## how to run this script
# x, y = prepare_dataset_for_modeling('us_census_income_data.csv')
