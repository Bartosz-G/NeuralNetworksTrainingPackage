import numpy as np
import pandas as pd
import torch
#from torcheval.metrics.functional import r2_score as r2_score_torch
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, r2_score, mean_squared_error



# ================== Depreciated ============================

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

        metrics['accuracy_score'] = float(accuracy_score(y_class, yhat_class))
        metrics['roc_auc_score'] = float(roc_auc_score(y_score, yhat_score, multi_class='ovo', average='macro'))
        metrics['confusion_matrix'] =[[int(x) for x in row] for row in confusion_matrix(y_class, yhat_class)]


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

        metrics['r2_score'] = float(r2_score(y, yhat))
        metrics['RMSE'] = float(mean_squared_error(y, yhat, squared=False))
        metrics['se_quant'] = ((yhat - y)**2).quantile([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]).to_dict()

    return metrics


# ==============================================================
# ===TODO: Write a pytorch function for handling metrics
# ==============================================================
# def calc_metrics_torch(y, yhat, is_categorical):
#     assert isinstance(y, torch.Tensor) and isinstance(yhat, torch.Tensor), "this function only handles torch.Tensors, for other dtypes use calc_metrics"
#
#
#
#     metrics = {}
#     if is_categorical:
#         y_score, yhat_score = y.detach(), yhat.detach()
#
#         y_true, y_pred = torch.argmax(y_score, dim=1), torch.argmax(yhat_score, dim=1)
#
#         # Number of classes
#         num_classes = y.shape[1]
#
#         # Create a confusion matrix
#         index_combinations = num_classes * y_true + y_pred
#         unique_vals, unique_counts = torch.unique(index_combinations, return_counts=True)
#
#         confusion_mat = torch.zeros(num_classes, num_classes, dtype=torch.int).to(y.device)
#
#         unique_counts = unique_counts.to(torch.int)  # Convert unique_counts to int dtype
#         confusion_mat[(unique_vals // num_classes), (unique_vals % num_classes)] = unique_counts
#
#         # Calculate accuracy
#         correct = torch.diag(confusion_mat).sum().item()
#         total = y.shape[0]
#         accuracy = correct / total
#
#         y_score, yhat_score = y_score.cpu().numpy(), yhat_score.cpu().numpy()
#         confusion_mat_list = confusion_mat.cpu().tolist()
#         metrics['accuracy_score'] = accuracy
#         metrics['roc_auc_score'] = roc_auc_score(y_score, yhat_score, multi_class='ovo', average='macro')
#         metrics['confusion_matrix'] = confusion_mat_list
#
#     else:
#         y, yhat = y.detach(), yhat.detach()
#
#         standard_errors = ((yhat - y) ** 2)
#         MSE = standard_errors.mean()
#         RMSE = torch.sqrt(MSE)
#         standard_errors_pd = pd.DataFrame(standard_errors.cpu().numpy())
#
#         metrics['r2_score'] = r2_score_torch(y, yhat).item()
#         metrics['RMSE'] = RMSE.item()
#         metrics['se_quant'] = standard_errors_pd.quantile([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]).to_dict()
#
#     return metrics


# ==============================================================
