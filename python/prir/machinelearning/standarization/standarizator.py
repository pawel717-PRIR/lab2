from sklearn import preprocessing


def standard_scaler(data):
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(data)


def min_max_scaler(data):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(data)
