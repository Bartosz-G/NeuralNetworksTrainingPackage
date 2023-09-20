import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import QuantileTransformer




# class exampleTransformationTemplate():
#     def __init__(self, transform = 'all'):
#         self.seed = None
#         self.parent = None
#         self.transform = transform
#
#     def apply(self, X, y, categorical_indicator, attribute_names):
#         if not self.seed:
#             self.seed = self.parent.seed
#
#
#         #
#
#         return X, y, categorical_indicator, attribute_names



# =============== Pre-processing functions to be applied before ===============

class truncateData():
    def __init__(self, n, seed = None, transform = 'all'):
        self.parent = None
        self.transform = transform

        self.n = n
        self.seed = seed

    def apply(self, X, y, categorical_indicator, attribute_names):
        if not self.seed:
            self.seed = self.parent.seed

        X, y = shuffle(X, y, random_state=self.seed)
        X, y = X.head(self.n), y.head(self.n)

        X,y = X.reset_index(drop=True), y.reset_index(drop=True)

        return X, y, categorical_indicator, attribute_names


class balancedTruncateData():
    def __init__(self, n, seed=None, transform='all'):
        self.parent = None
        self.transform = transform

        self.n = n
        self.seed = seed

    def apply(self, X, y, categorical_indicator, attribute_names):
        if X.shape[0] < self.n:
            return X, y, categorical_indicator, attribute_names

        if not self.seed:
            self.seed = self.parent.seed

        # Ensure y_series is a pandas Series
        if isinstance(y, pd.Series):
            y_series = y
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] > 1:
                y_series = y.idxmax(axis=1)
            else:
                y_series = pd.Series(y.values.ravel())

        # Get class counts and identify minority classes
        class_counts = y_series.value_counts()
        num_classes = len(class_counts)
        min_samples_per_class = self.n // num_classes

        # Gathering indices for a balanced dataset
        balanced_indices = []
        for class_label in class_counts.index:
            class_indices = y_series[y_series == class_label].index.tolist()

            if class_counts[class_label] < min_samples_per_class:
                balanced_indices.extend(class_indices)
            else:
                balanced_indices.extend(np.random.choice(class_indices, min_samples_per_class, replace=False))

        # Ensuring the number of samples does not exceed self.n
        balanced_indices = balanced_indices[:self.n]

        # Shuffle the balanced indices to get a random sample
        np.random.seed(self.seed)
        np.random.shuffle(balanced_indices)

        # Getting the balanced dataset
        X_balanced = X.loc[balanced_indices].reset_index(drop=True)
        y_balanced = y.loc[balanced_indices].reset_index(drop=True)


        return X_balanced, y_balanced, categorical_indicator, attribute_names




class filterCardinality():
    def __init__(self, transform = 'all'):
        self.parent = None
        self.transform = transform

        self.numeric_that_should_be_categorical = 2
        self.too_high_cardinality = 20
        self.not_enough_numeric = 10


    def apply(self, X, y, categorical_indicator, attribute_names):
        valid_cols = []
        categorical_indicator_filtered = []
        attribute_names_filtered = []
        for colname, nunique, iscat, name in zip(attribute_names, X.apply(pd.Series.nunique).values, categorical_indicator, attribute_names):

            if iscat:
                # filtering out high cardinality categorical
                if nunique <= self.too_high_cardinality:
                    valid_cols.append(colname)
                    categorical_indicator_filtered.append(iscat)
                    attribute_names_filtered.append(name)

            # converting low numeric to categorical
            elif nunique <= self.numeric_that_should_be_categorical:
                valid_cols.append(colname)
                categorical_indicator_filtered.append(True)
                attribute_names_filtered.append(name)

            # only keeping high cardinality numeric
            elif nunique >= self.not_enough_numeric:
                valid_cols.append(colname)
                categorical_indicator_filtered.append(False)
                attribute_names_filtered.append(name)


        return X[valid_cols], y, categorical_indicator_filtered, attribute_names_filtered



class quantileTransform():
    def __init__(self,n_quantiles=1000, output_distribution='uniform',
                 ignore_implicit_zeros=False,
                 subsample=10000,
                 random_state=None,
                 copy=True,
                 transform = 'all'):
        self.parent = None
        self.transform = transform


        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy

    def apply(self, X, y, categorical_indicator, attribute_names):
        if not self.random_state:
            self.random_state = self.parent.seed

        if all(categorical_indicator):
            return X, y, categorical_indicator, attribute_names

        qt = QuantileTransformer(n_quantiles=self.n_quantiles,
                                      output_distribution=self.output_distribution,
                                      ignore_implicit_zeros=self.ignore_implicit_zeros,
                                      subsample=self.subsample,
                                      random_state=self.random_state,
                                      copy=self.copy)

        categorical_np = np.array(categorical_indicator)

        X.loc[:, ~categorical_np] = qt.fit_transform(X.loc[:, ~categorical_np])

        return X, y, categorical_indicator, attribute_names



class toDataFrame():
    def __init__(self, transform = 'all'):
        self.parent = None
        self.transform = transform

    def apply(self, X, y, categorical_indicator, attribute_names):
        assert isinstance(X, (pd.Series, pd.DataFrame)), "X must be a Pandas Series or DataFrame"
        assert isinstance(y, (pd.Series, pd.DataFrame)), "Y must be a Pandas Series or DataFrame"

        if not isinstance(X, pd.DataFrame):
            X = X.to_frame()
        if not isinstance(y, pd.DataFrame):
            y = y.to_frame()

        return X, y, categorical_indicator, attribute_names


class oneHotEncodePredictors():
    def __init__(self, transform = 'all'):
        self.parent = None
        self.transform = transform

    def apply(self, X, y, categorical_indicator, attribute_names):
        X_dummies = pd.get_dummies(X, columns=X.columns[categorical_indicator])
        X_dummies = X_dummies.astype(float)

        new_attribute_names = list(X_dummies.columns)

        original_categorical_columns = X.columns[categorical_indicator]

        new_categorical_indicator = [
            any(new_col.startswith(orig_cat_col + "_") for orig_cat_col in original_categorical_columns)
            for new_col in new_attribute_names
        ]

        return X_dummies, y, new_categorical_indicator, new_attribute_names

class oneHotEncodeTargets():
    def __init__(self, transform = 'all'):
        self.parent = None
        self.transform = transform

    def apply(self, X, y, categorical_indicator, attribute_names):

        y = pd.get_dummies(y, dtype=int)

        # if isinstance(y, pd.DataFrame):
        #     is_categorical = any(y[col].dtype.name == 'category' for col in y.columns)
        #     if is_categorical:
        #         y = pd.get_dummies(y)
        #
        # if isinstance(y, pd.Series):
        #     is_categorical = y.dtype.name == 'category'
        #
        #     if is_categorical:
        #         y = pd.get_dummies(y)

        return X, y, categorical_indicator, attribute_names


class splitTrainValTest():
    def __init__(self, split = [0.5, 0.25, 0.25]):
        # Prevents the parent from passing X, y, categorical_indicator, attribute_names, and gets kwargs instead
        self.special = True

        self.parent = None
        self.split = split

    def apply(self, **kwargs):
        assert self.parent.val is None, "Tried splitting into train, val, test but validation already exists"
        assert self.parent.test is None, "Tried splitting into train, val, test but test already exists"


        X, y, categorical_indicator, attribute_names = self.parent.train

        data_nrow = len(X)
        train_count, val_count = int(data_nrow * self.split[0]), int(data_nrow * self.split[1])
        test_count = data_nrow - train_count - val_count

        shuffled_indices = np.random.permutation(data_nrow)

        train_indices, val_indices, test_indices = shuffled_indices[:train_count], shuffled_indices[train_count:train_count + val_count], shuffled_indices[train_count + val_count:]

        train_data = (X.iloc[train_indices].reset_index(drop=True), y.iloc[train_indices].reset_index(drop=True),
                      categorical_indicator, attribute_names)
        val_data = (X.iloc[val_indices].reset_index(drop=True), y.iloc[val_indices].reset_index(drop=True),
                    categorical_indicator,attribute_names)
        test_data = (X.iloc[test_indices].reset_index(drop=True), y.iloc[test_indices].reset_index(drop=True),
                     categorical_indicator,attribute_names)

        self.parent.train = train_data
        self.parent.val = val_data
        self.parent.test = test_data


class balancedSplitTrainValTest():
    def __init__(self, split = [0.5, 0.25, 0.25]):
        # Prevents the parent from passing X, y, categorical_indicator, attribute_names, and gets kwargs instead
        self.special = True

        self.parent = None
        self.split = split

    def apply(self, **kwargs):
        assert self.parent.val is None, "Tried splitting into train, val, test but validation already exists"
        assert self.parent.test is None, "Tried splitting into train, val, test but test already exists"

        X, y, categorical_indicator, attribute_names = self.parent.train

        # Ensure y_series is a pandas Series
        if isinstance(y, pd.Series):
            y_series = y
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] > 1:
                y_series = y.idxmax(axis=1)
            else:
                y_series = pd.Series(y.values.ravel())

        unique_classes = y_series.unique()
        class_indices = [np.where(y_series == uc)[0] for uc in unique_classes]

        train_indices, val_indices, test_indices = [], [], []

        for indices in class_indices:
            np.random.shuffle(indices)
            train_size = int(len(indices) * self.split[0])
            val_size = int(len(indices) * self.split[1])

            train_indices.extend(indices[:train_size])
            val_indices.extend(indices[train_size:train_size + val_size])
            test_indices.extend(indices[train_size + val_size:])

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        train_data = (X.iloc[train_indices].reset_index(drop=True), y.iloc[train_indices].reset_index(drop=True),
                      categorical_indicator, attribute_names)
        val_data = (
        X.iloc[val_indices].reset_index(drop=True), y.iloc[val_indices].reset_index(drop=True), categorical_indicator,
        attribute_names)
        test_data = (
        X.iloc[test_indices].reset_index(drop=True), y.iloc[test_indices].reset_index(drop=True), categorical_indicator,
        attribute_names)

        self.parent.train = train_data
        self.parent.val = val_data
        self.parent.test = test_data





# ============= Pytorch Specific transformations and objects =====================
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, categorical_indicator, attribute_names, tensor_type=torch.float):
        assert isinstance(X, pd.DataFrame), "X must be a Pandas DataFrame"
        assert isinstance(Y, pd.DataFrame), "Y must be a Pandas DataFrame"

        self.X, self.Y = X, Y
        self.categorical_indicator = categorical_indicator
        self.attribute_names = attribute_names

        assert isinstance(tensor_type, torch.dtype), "tensor_type must be a valid torch.dtype"
        self.tensor_type = tensor_type

    def get_dims(self):
        num_columns_X = self.X.shape[1]
        num_columns_Y = self.Y.shape[1] if isinstance(self.Y, pd.DataFrame) else 1
        return {'input_dim': num_columns_X, 'output_dim':num_columns_Y}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X.iloc[[idx], :], self.Y.iloc[[idx], :]

        x, y = torch.tensor(x.values.squeeze(axis=0), dtype=self.tensor_type), torch.tensor(y.values.squeeze(axis=0), dtype=self.tensor_type)

        return x, y


class CustomDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, train_dataset, relative_indices):
        assert isinstance(train_dataset, CustomDataset), "train_dataset must be an instance of CustomDataset"
        assert isinstance(relative_indices,
                          np.ndarray) and relative_indices.ndim == 1, "Relative indices must be a 1D NumPy array"
        assert len(relative_indices) <= len(
            train_dataset), "Length of relative_indices must not be greater than the length of train_dataset"

        self.train_dataset = train_dataset
        self.relative_indices = relative_indices

    def set_new_indices(self, new_relative_indices):
        assert isinstance(new_relative_indices,
                          np.ndarray) and new_relative_indices.ndim == 1, "Relative indices must be a 1D NumPy array"
        assert len(new_relative_indices) <= len(
            self.train_dataset), "Length of relative_indices must not be greater than the length of train_dataset"
        self.relative_indices = new_relative_indices

    def __len__(self):
        return len(self.relative_indices)

    def __getitem__(self, idx):
        absolute_index = self.relative_indices[idx]

        return self.train_dataset[absolute_index]


class toPyTorchDatasets():
    def __init__(self, wrapper = CustomDataset):
        self.special = True
        self.parent = None
        self.wrapper = wrapper

    def apply(self, **kwargs):
        X, y, categorical_indicator, attribute_names = self.parent.train
        self.parent.train = self.wrapper(X, y, categorical_indicator, attribute_names)

        if self.parent.val is not None:
            X, y, categorical_indicator, attribute_names = self.parent.val
            self.parent.val = self.wrapper(X, y, categorical_indicator, attribute_names)

        if self.parent.test is not None:
            X, y, categorical_indicator, attribute_names = self.parent.test
            self.parent.test = self.wrapper(X, y, categorical_indicator, attribute_names)





# ================== Depreciated ============================
def get_train_test(X, y, categorical_indicator, attribute_names, data_pre_processing,
                   task = 'regression',
                   model = None ,
                   train_split = 0.8,
                   args = None):
    """Processes dataset
    expects the results from `opml_load_task`, 0<= train_spli <= 1, seed
    returns train CustomDataset Object, test CustomDataset Object, input_dim, output_dim
    """

    assert 0 <= train_split <= 1, "train_split must be between 0 and 1."


    assert isinstance(X, (pd.Series, pd.DataFrame)), "X must be a Pandas Series or DataFrame"
    assert isinstance(y, (pd.Series, pd.DataFrame)), "Y must be a Pandas Series or DataFrame"

    X, y, categorical_indicator, attribute_names = data_pre_processing.apply(task, X, y, categorical_indicator, attribute_names)
    X, y, categorical_indicator, attribute_names = data_pre_processing.apply(model, X, y, categorical_indicator, attribute_names)

    data_nrows = len(X)
    num_train_samples = int(data_nrows * train_split)

    train_indices = np.random.choice(X.index, num_train_samples, replace=False)
    test_indices = np.setdiff1d(X.index, train_indices)

    # Only for LCN
    if args:
        num_columns_X = X.shape[1]
        num_columns_y = y.shape[1] if isinstance(y, pd.DataFrame) else 1
        args.set_input_dims(num_columns_X, num_columns_y)

    return CustomDataset(X, y, train_indices), CustomDataset(X, y, test_indices)


def get_train_val_test(X, y, categorical_indicator, attribute_names,
                       data_pre_processing,
                       task = 'regression',
                       model = None ,
                       split = [0.5, 0.25, 0.25],
                       args = None):


    assert isinstance(X, (pd.Series, pd.DataFrame)), "X must be a Pandas Series or DataFrame"
    assert isinstance(y, (pd.Series, pd.DataFrame)), "Y must be a Pandas Series or DataFrame"

    X, y, categorical_indicator, attribute_names = data_pre_processing.apply(task, X, y, categorical_indicator, attribute_names)
    X, y, categorical_indicator, attribute_names = data_pre_processing.apply(model, X, y, categorical_indicator, attribute_names)

    data_nrow = len(X)
    train_count, val_count = int(data_nrow * split[0]), int(data_nrow * split[1])
    test_count = data_nrow - train_count - val_count

    shuffled_indices = np.random.permutation(data_nrow)

    train_indices, val_indices, test_indices = shuffled_indices[:train_count], shuffled_indices[train_count:train_count + val_count], shuffled_indices[train_count + val_count:]

    # Only for LCN
    if args:
        num_columns_X = X.shape[1]
        num_columns_y = y.shape[1] if isinstance(y, pd.DataFrame) else 1
        args.set_input_dims(num_columns_X, num_columns_y)

    return CustomDataset(X, y, train_indices), CustomDataset(X, y, val_indices), CustomDataset(X, y, test_indices)




class kfold_dataloader_iterator():
    def __init__(self,
                 dataset,
                 n_splits=10,  # kfold arguments
                 random_state=42,
                 batch_size=16,  # data_loader argument
                 shuffle_kfold=True,
                 shuffle_dataloader=True):
        assert isinstance(dataset, CustomDataset), "train_dataset must be an instance of CustomDataset"
        self.__dataset = dataset
        self.__kf = KFold(n_splits=n_splits, shuffle=shuffle_kfold, random_state=42)
        self.__kf_iter = self.__kf.split(list(range(len(self.__dataset))))

        assert isinstance(batch_size, int), "Batch size must be an int"
        self.batch_size = batch_size
        self.shuffle_dataloader = shuffle_dataloader

        self.train_data = None
        self.val_data = None

    def __iter__(self):
        return self

    def __next__(self):
        train_indices, val_indices = next(self.__kf_iter)

        self.train_data = CustomDatasetWrapper(self.__dataset, train_indices)
        self.val_data = CustomDatasetWrapper(self.__dataset, val_indices)

        train_dataloader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size,
                                                       shuffle=self.shuffle_dataloader)
        val_dataloader = torch.utils.data.DataLoader(self.val_data, batch_size=len(self.val_data), shuffle=False)

        return train_dataloader, val_dataloader




if __name__ == "__main__":

    print('==== Testing CustomDataset ====')
    test_x = pd.DataFrame({
        'a': [1, 2, 3, 4, 5, 6],
        'b': [4, 5, 6, 8, 1, 9]
    })

    test_y = pd.DataFrame({
        'a': [1, 2, 3, 4, 5, 6]
    })

    indices = np.array([2, 3, 4])

    test_obj = CustomDataset(test_x, test_y, indices)
    print(test_obj[0])
    print(test_obj[0][1])
    print(test_obj[0][1].shape)
    print(torch.tensor([3], dtype=torch.float).shape)



    assert torch.equal(test_obj[0][0], torch.tensor([3, 6], dtype=torch.float)) and torch.equal(test_obj[0][1],
                                                                                                  torch.tensor([3],
                                                                                                               dtype=torch.float)), 'test failed'
    print('test passed')
    assert torch.equal(test_obj[1][0], torch.tensor([4, 8], dtype=torch.float)) and torch.equal(test_obj[1][1],
                                                                                                  torch.tensor([4],
                                                                                                               dtype=torch.float)), 'test failed'
    print('test passed')
    assert torch.equal(test_obj[2][0], torch.tensor([5, 1], dtype=torch.float)) and torch.equal(test_obj[2][1],
                                                                                                  torch.tensor([5],
                                                                                                               dtype=torch.float)), 'test failed'
    print('test passed')
    assert torch.equal(test_obj[2][0], torch.tensor([5, 1], dtype=torch.float)) and torch.equal(test_obj[2][1],
                                                                                                  torch.tensor([5],
                                                                                                               dtype=torch.float)), 'test failed'
    print('test passed')
    assert len(test_obj) == len(indices), 'test failed'
    print('test passed')

    # Concatenating the inputs (assuming they are 2D tensors)
    concatenated_X = torch.cat([test_obj[i][0] for i in range(len(test_obj))], dim=0)
    concatenated_Y = torch.cat([test_obj[i][1] for i in range(len(test_obj))], dim=0)

    # Expected concatenated tensors
    expected_X = torch.cat([torch.tensor([3, 6], dtype=torch.float),
                            torch.tensor([4, 8], dtype=torch.float),
                            torch.tensor([5, 1], dtype=torch.float)], dim=0)
    expected_Y = torch.cat([torch.tensor([3], dtype=torch.float),
                            torch.tensor([4], dtype=torch.float),
                            torch.tensor([5], dtype=torch.float)], dim=0)

    # Check if the concatenated tensors are equal to the expected tensors
    assert torch.equal(concatenated_X, expected_X), 'Concatenated X test failed'
    assert torch.equal(concatenated_Y, expected_Y), 'Concatenated Y test failed'
    print('Concatenation test passed')

    print('==== Testing CustomDatasetWrapper ====')

    test_x = pd.DataFrame({
        'a': [1, 2, 3, 4, 5, 6],
        'b': [4, 5, 6, 8, 1, 9]
    })

    test_y = pd.DataFrame({
        'a': [1, 2, 3, 4, 5, 6]
    })

    indices_1 = np.array([2, 3, 4])
    indices_2 = np.array([1, 2])

    test_obj = CustomDataset(test_x, test_y, indices_1)
    test_outer_obj = CustomDatasetWrapper(test_obj, indices_2)

    # Testing index [0]
    assert torch.equal(test_outer_obj[0][0], torch.tensor([4, 8], dtype=torch.float)) and torch.equal(
        test_outer_obj[0][1], torch.tensor([4], dtype=torch.float)), 'test failed'
    print('test passed')

    # Testing index [1]
    assert torch.equal(test_outer_obj[1][0], torch.tensor([5, 1], dtype=torch.float)) and torch.equal(
        test_outer_obj[1][1], torch.tensor([5], dtype=torch.float)), 'test failed'
    print('test passed')

    # Concatenating the inputs (assuming they are 2D tensors)
    concatenated_X = torch.cat([test_outer_obj[i][0] for i in range(len(test_outer_obj))], dim=0)
    concatenated_Y = torch.cat([test_outer_obj[i][1] for i in range(len(test_outer_obj))], dim=0)

    # Expected concatenated tensors
    expected_X = torch.cat([torch.tensor([4, 8], dtype=torch.float),
                            torch.tensor([5, 1], dtype=torch.float)], dim=0)
    expected_Y = torch.cat([torch.tensor([4], dtype=torch.float),
                            torch.tensor([5], dtype=torch.float)], dim=0)

    # Check if the concatenated tensors are equal to the expected tensors
    assert torch.equal(concatenated_X, expected_X), 'Concatenated X test failed'
    assert torch.equal(concatenated_Y, expected_Y), 'Concatenated Y test failed'
    print('Concatenation test passed')

    print('=== test 2 ====')
    # New relative indices
    new_indices = np.array([0, 2])

    # Set new relative indices
    test_outer_obj.set_new_indices(new_indices)

    # Testing index [0] with new relative indices
    assert torch.equal(test_outer_obj[0][0], torch.tensor([3, 6], dtype=torch.float)) and torch.equal(
        test_outer_obj[0][1], torch.tensor([3], dtype=torch.float)), 'test failed'
    print('test passed')

    # Testing index [1] with new relative indices
    assert torch.equal(test_outer_obj[1][0], torch.tensor([5, 1], dtype=torch.float)) and torch.equal(
        test_outer_obj[1][1], torch.tensor([5], dtype=torch.float)), 'test failed'
    print('test passed')

    # Concatenating the inputs (assuming they are 2D tensors) with new relative indices
    concatenated_X = torch.cat([test_outer_obj[i][0] for i in range(len(test_outer_obj))], dim=0)
    concatenated_Y = torch.cat([test_outer_obj[i][1] for i in range(len(test_outer_obj))], dim=0)

    # Expected concatenated tensors with new relative indices
    expected_X = torch.cat([torch.tensor([3, 6], dtype=torch.float),
                            torch.tensor([5, 1], dtype=torch.float)], dim=0)
    expected_Y = torch.cat([torch.tensor([3], dtype=torch.float),
                            torch.tensor([5], dtype=torch.float)], dim=0)

    # Check if the concatenated tensors are equal to the expected tensors
    assert torch.equal(concatenated_X, expected_X), 'Concatenated X test failed'
    assert torch.equal(concatenated_Y, expected_Y), 'Concatenated Y test failed'
    print('Concatenation test passed')


    print('==== Testing kfold ====')

    test_x = pd.DataFrame({
        'a': list(range(1, 21)),
        'b': list(range(4, 24))
    })

    test_y = pd.DataFrame({
        'a': list(range(1, 21))
    })

    indices_1 = np.array(range(test_x[test_x.columns[0]].count()))
    print(f'lenght: {len(test_x)}')

    # Create a CustomDataset object
    test_obj = CustomDataset(test_x, test_y, indices_1)

    # Create kfold_dataloader_iterator object
    kfold_iterator = kfold_dataloader_iterator(test_obj, n_splits=4, batch_size=4)

    # Iterate through the kfold splits and perform tests
    for i, (train_dataloader, val_dataloader) in enumerate(kfold_iterator):
        print(f"Fold {i + 1}:")

        # Check the length of the training and validation dataloaders
        print(f'train_dataloader.dataset = {len(train_dataloader.dataset)}')
        assert len(train_dataloader.dataset) == 15, f"Training data length in fold {i + 1} is incorrect"
        print(f'val_dataloader.dataset = {len(val_dataloader.dataset)}')
        assert len(val_dataloader.dataset) == 5, f"Validation data length in fold {i + 1} is incorrect"

        print(f'Data: {next(iter(train_dataloader))}')

    # Check that the iteration has indeed ended
    try:
        next(kfold_iterator)
    except StopIteration:
        print("Iteration has ended as expected.")
    else:
        print("Unexpectedly, the iteration has not ended.")