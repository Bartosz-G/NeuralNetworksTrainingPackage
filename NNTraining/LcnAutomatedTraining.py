import numpy as np
import pandas as pd
import torch
from torcheval.metrics.functional import r2_score as r2_score_torch
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.preprocessing import QuantileTransformer



class Hyperparams:
    def __init__(self, depth, seed, drop_type, p, ensemble_n, shrinkage, back_n, net_type, hidden_dim, anneal,
                 optimizer, batch_size, epochs, lr, momentum, no_cuda, lr_step_size, gamma, task):
        self.depth = depth
        self.seed = seed
        self.drop_type = drop_type
        self.p = p
        self.ensemble_n = ensemble_n
        self.shrinkage = shrinkage
        self.back_n = back_n
        self.net_type = net_type
        self.hidden_dim = hidden_dim
        self.anneal = anneal
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.no_cuda = no_cuda
        self.lr_step_size = lr_step_size
        self.gamma = gamma
        self.task = task
        self.input_dim = None
        self.output_dim = None

    def __bool__(self):
        return True

    def set_input_dims(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, relative_indices, tensor_type=torch.float):
        assert isinstance(X, pd.DataFrame), "X must be a Pandas DataFrame"
        assert isinstance(Y, pd.DataFrame), "Y must be a Pandas DataFrame"

        self.X, self.Y = X, Y

        assert isinstance(relative_indices, np.ndarray) and relative_indices.ndim == 1, "Relative indices must be a 1D NumPy array"
        self.relative_indices = np.sort(relative_indices)

        assert isinstance(tensor_type, torch.dtype), "tensor_type must be a valid torch.dtype"
        self.tensor_type = tensor_type

    def __len__(self):
        return len(self.relative_indices)

    def __getitem__(self, idx):
        absolute_index = self.relative_indices[idx]


        x, y = self.X.iloc[[absolute_index], :], self.Y.iloc[[absolute_index], :]

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



def get_train_test(X, y, categorical_indicator, attribute_names, train_split = 0.8,
                   trunctuate = 20000, seed = None, args = None):
    """Processes dataset
    expects the results from `opml_load_task`, 0<= train_spli <= 1, seed
    returns train CustomDataset Object, test CustomDataset Object, input_dim, output_dim
    """

    assert 0 <= train_split <= 1, "train_split must be between 0 and 1."

    # ==============================================================
    # ===TODO: Refactor as a seperate, preprocessing function
    # ==============================================================
    assert isinstance(X, (pd.Series, pd.DataFrame)), "X must be a Pandas Series or DataFrame"
    assert isinstance(y, (pd.Series, pd.DataFrame)), "Y must be a Pandas Series or DataFrame"

    # itterative filtering and processing
    valid_cols = []
    categorical_indicator_filtered = []
    for colname, nunique, iscat in zip(attribute_names, X.apply(pd.Series.nunique).values, categorical_indicator):

        if iscat:
            # filtering out high cardinality categorical
            if nunique <= 20:
                valid_cols.append(colname)
                categorical_indicator_filtered.append(iscat)

        # converting low numeric to categorical
        elif nunique <= 2:
            valid_cols.append(colname)
            categorical_indicator_filtered.append(True)

        # only keeping high cardinality numeric
        elif nunique >= 10:
            valid_cols.append(colname)
            categorical_indicator_filtered.append(False)

        # normalizing numeric columns with quantile distribution
        if not iscat:
            qt = QuantileTransformer(random_state=seed)
            X[colname] = qt.fit_transform(X[[colname]])

    # only keeping valid columns
    X = X[valid_cols]

    if not isinstance(X, pd.DataFrame):
        X = X.to_frame()
    if not isinstance(y, pd.DataFrame):
        y = y.to_frame()

    if seed:
        np.random.seed(seed)

    if trunctuate:
        X, y = shuffle(X, y, random_state=seed)
        X, y = X.head(trunctuate), y.head(trunctuate)



    # Fixed to one-hot encoding for all categorical variables
    X = pd.get_dummies(X, X.columns[categorical_indicator_filtered])

    is_categorical = any(y[col].dtype.name == 'category' for col in y.columns)
    if is_categorical:
        y = pd.get_dummies(y)
    # ==============================================================

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
                       split = [0.5, 0.25, 0.25],
                       trunctuate = 20000,
                       seed = None,
                       args = None):

    assert isinstance(split, list) and len(split) == 3 and all(isinstance(x, float) for x in split), "split must be a list of floats, for train, val, test e.g. [0.5, 0.25, 0.25]"
    assert isinstance(trunctuate, int) or trunctuate is None or trunctuate is False, "trunctuate must be a whole number or either a False or None"

    # ==============================================================
    # ===TODO: Refactor as a seperate, preprocessing function
    # ==============================================================
    assert isinstance(X, (pd.Series, pd.DataFrame)), "X must be a Pandas Series or DataFrame"
    assert isinstance(y, (pd.Series, pd.DataFrame)), "Y must be a Pandas Series or DataFrame"

    # itterative filtering and processing
    valid_cols = []
    categorical_indicator_filtered = []
    for colname, nunique, iscat in zip(attribute_names, X.apply(pd.Series.nunique).values, categorical_indicator):

        if iscat:
            # filtering out high cardinality categorical
            if nunique <= 20:
                valid_cols.append(colname)
                categorical_indicator_filtered.append(iscat)

        # converting low numeric to categorical
        elif nunique <= 2:
            valid_cols.append(colname)
            categorical_indicator_filtered.append(True)

        # only keeping high cardinality numeric
        elif nunique >= 10:
            valid_cols.append(colname)
            categorical_indicator_filtered.append(False)

        # normalizing numeric columns with quantile distribution
        if not iscat:
            qt = QuantileTransformer(random_state=seed)
            X[colname] = qt.fit_transform(X[[colname]])

    # only keeping valid columns
    X = X[valid_cols]

    if not isinstance(X, pd.DataFrame):
        X = X.to_frame()
    if not isinstance(y, pd.DataFrame):
        y = y.to_frame()

    if seed:
        np.random.seed(seed)

    if trunctuate:
        X, y = shuffle(X, y, random_state=seed)
        X, y = X.head(trunctuate), y.head(trunctuate)



    # Fixed to one-hot encoding for all categorical variables
    X = pd.get_dummies(X, X.columns[categorical_indicator_filtered])

    is_categorical = any(y[col].dtype.name == 'category' for col in y.columns)
    if is_categorical:
        y = pd.get_dummies(y)
    # ==============================================================


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


# ========== Metric Calculations =============

def np_get_cls_score(y, yhat):
    # ==============================================================
    # ===TODO: Add multi-class handling
    # ==============================================================

    # Expects np.ndarray
    assert y.shape == yhat.shape, f"expects same dimensions, received y:{y.shape}, yhat: {yhat.shape}"
    assert y.shape[-1] >= 2 and yhat.shape[-1] >= 2, f"for classification expects score metrics, not binary classifications"

    y_class = np.argmax(y, axis=1)
    yhat_class = np.argmax(yhat, axis=1)

    y_score = y.copy()
    yhat_score = yhat.copy()

    return y_class, yhat_class, y_score, yhat_score


def tensor_get_cls_score(y, yhat):
    # ==============================================================
    # ===TODO: Add multi-class handling
    # ==============================================================

    # Expects torch.Tensor
    assert y.shape == yhat.shape, f"expects same dimensions, received y:{y.shape}, yhat: {yhat.shape}"
    assert y.shape[-1] >= 2 and yhat.shape[-1] >= 2, f"for classification expects score metrics, not binary classifications"

    y_class = torch.argmax(y, dim=1).detach().cpu().numpy()
    yhat_class = yhat.max(1, keepdim=True)[1].detach().cpu().numpy()

    y_score = y.detach().cpu().numpy()
    yhat_score = yhat.detach().cpu().numpy()

    return y_class, yhat_class, y_score, yhat_score


def pd_get_cls_score(y, yhat):
    # ==============================================================
    # ===TODO: Add multi-class handling
    # ==============================================================

    # expects pd.DataFrame
    assert y.shape == yhat.shape, f"expects same dimensions, received y:{y.shape}, yhat: {yhat.shape}"
    assert y.shape[-1] >= 2 and yhat.shape[-1] >= 2, f"for classification expects score metrics, not binary classifications"

    y_class = np.argmax(y.values, axis=1)
    yhat_class = np.argmax(yhat.values, axis=1)

    y_score = y.values.copy()
    yhat_score = yhat.values.copy()

    return y_class, yhat_class, y_score, yhat_score


def calc_metrics(y, yhat, is_categorical):
    # Check if inputs are NumPy arrays or torch tensors, and convert to DataFrame

    metrics = {}
    if is_categorical:
        if isinstance(y, np.ndarray) and isinstance(yhat, np.ndarray):
            y_class, yhat_class, y_score, yhat_score = np_get_cls_score(y, yhat)
        elif isinstance(y, torch.Tensor) and isinstance(yhat, torch.Tensor):
            y_class, yhat_class, y_score, yhat_score = tensor_get_cls_score(y, yhat)
        elif isinstance(y, pd.DataFrame) and isinstance(yhat, pd.DataFrame):
            y_class, yhat_class, y_score, yhat_score = pd_get_cls_score(y, yhat)
        else:
            raise TypeError("Type missmatch between y and yhat, both need to be one of type: np.ndarray, torch.Tensor, pd.DataFrame")

        # ==============================================================
        # ===TODO: Refactor as a seperate, metrics function
        # ==============================================================

        metrics['accuracy_score'] = accuracy_score(y_class, yhat_class)
        metrics['roc_auc_score'] = roc_auc_score(y_score, yhat_score, multi_class='ovo', average='macro')
        metrics['confusion_matrix'] = [list(r) for r in confusion_matrix(y_class, yhat_class)]

    else:
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
        elif isinstance(y, torch.Tensor):
            y = pd.DataFrame(y.detach().cpu().numpy())
        elif not isinstance(y, pd.DataFrame):
            raise AssertionError("Allowed data types of y and yhat are: np.ndarray, torch.Tensor, pd.DataFrame")

        if isinstance(yhat, np.ndarray):
            yhat = pd.DataFrame(yhat)
        elif isinstance(yhat, torch.Tensor):
            yhat = pd.DataFrame(yhat.detach().cpu().numpy())
        elif not isinstance(yhat, pd.DataFrame):
            raise AssertionError("Allowed data types of y and yhat are: np.ndarray, torch.Tensor, pd.DataFrame")

        # ==============================================================
        # ===TODO: Refactor as a seperate, metrics function
        # ==============================================================

        metrics['r2_score'] = r2_score(y, yhat)
        metrics['RMSE'] = mean_squared_error(y, yhat, squared=False)
        metrics['se_quant'] = ((yhat - y)**2).quantile([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]).to_dict()

    return metrics


# ==============================================================
# ===TODO: Write a pytorch function for handling metrics
# ==============================================================
def calc_metrics_torch(y, yhat, is_categorical):
    assert isinstance(y, torch.Tensor) and isinstance(yhat, torch.Tensor), "this function only handles torch.Tensors, for other dtypes use calc_metrics"



    metrics = {}
    if is_categorical:
        y_score, yhat_score = y.detach(), yhat.detach()

        y_true, y_pred = torch.argmax(y_score, dim=1), torch.argmax(yhat_score, dim=1)

        # Number of classes
        num_classes = y.shape[1]

        # Create a confusion matrix
        index_combinations = num_classes * y_true + y_pred
        unique_vals, unique_counts = torch.unique(index_combinations, return_counts=True)

        confusion_mat = torch.zeros(num_classes, num_classes, dtype=torch.int)

        unique_counts = unique_counts.to(torch.int)  # Convert unique_counts to int dtype
        confusion_mat[(unique_vals // num_classes), (unique_vals % num_classes)] = unique_counts

        # Calculate accuracy
        correct = torch.diag(confusion_mat).sum().item()
        total = y.shape[0]
        accuracy = correct / total

        y_score, yhat_score = y_score.cpu().numpy(), yhat_score.cpu().numpy()
        metrics['accuracy_score'] = accuracy
        metrics['roc_auc_score'] = roc_auc_score(y_score, yhat_score, multi_class='ovo', average='macro')
        metrics['confusion_matrix'] = confusion_mat

    else:
        y, yhat = y.detach(), yhat.detach()

        standard_errors = ((yhat - y) ** 2)
        MSE = standard_errors.mean()
        RMSE = torch.sqrt(MSE)
        standard_errors_pd = pd.DataFrame(standard_errors.cpu().numpy())

        metrics['r2_score'] = r2_score_torch(y, yhat).item()
        metrics['RMSE'] = RMSE.item()
        metrics['se_quant'] = standard_errors_pd.quantile([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]).to_dict()


# ==============================================================

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