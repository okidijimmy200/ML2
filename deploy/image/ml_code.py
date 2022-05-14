import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import model_selection, preprocessing
from sklearn.mixture import BayesianGaussianMixture

import torch
import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (
    Dropout,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential,
    Conv2d,
    ConvTranspose2d,
    BatchNorm2d,
    Sigmoid,
    init,
    BCELoss,
    CrossEntropyLoss,
    SmoothL1Loss,
)
from tqdm import tqdm
import warnings


class DataPreprocessing:
    """
    Description of class members:
    1. self.data: pd.DataFrame - Stores the training_data
    2. self.categorical_columns: list - Stores list of categorical columns in the training data
    3. self.log_columns: list - Stores list of log columns in the training data
    4. self.integer_columns: list - Stores list of integer columns in the training data
    5. self.mixed_columns: dict - Stores mixed columns in the training data as a dictionary with keys as the column names and values as the categorical modes in the mixed column
    6. self.lower_bounds: dict - Stored minimum values of log-columns with column name as key and minimum value as value
    7. self.label_encoder_list: list - Stores the label-encoders used for encoding categorical columns
    8. self.column_types: dict
    9. self.inverse_transformed_data: pd.DataFrame - Store inverse transformed data from the synthetic data after reversing all pre-processing methods
    """

    def __init__(self, data: pd.DataFrame, categorical_cols: list, log_cols: list, mixed_cols: dict, integer_cols: list, problem_type: dict):

        self.data = data
        self.lower_bounds = {}
        self.label_encoder_list = []
        self.column_types = {}
        self.column_types["categorical"] = []
        self.column_types["mixed"] = {}

        # Saving column types - Categorical, Log, Mixed & Integer
        self._assign_columns(categorical_cols, log_cols, mixed_cols, integer_cols)

        # Handling missing values
        self._process_missing_values()

        # Transforming log columns
        if self.log_columns:
            self._process_log_columns()

        # Encoding categorical columns
        self._encode_categorical_columns()

        super().__init__()

    """
  Saving different types of columns - Categorical, Log, Mixed & Integer as members of the DataPreprocessing class
  """

    def _assign_columns(self, categorical_cols: list, log_cols: list, mixed_cols: dict, integer_cols: list):
        self.categorical_columns = categorical_cols
        self.log_columns = log_cols
        self.mixed_columns = mixed_cols
        self.integer_columns = integer_cols

    """
  Replace missing values with a dummy value: -9999999, thereby causing following changes:
  1. Adds another categorical mode to mixed columns
  2. Converts continuous columns into mixed columns with 1 categorical mode
  """

    def _process_missing_values(self):

        # Replacing missing values
        self.data = self.data.replace(r" ", np.nan)
        self.data = self.data.fillna("empty")

        all_columns = set(self.data.columns)

        # Categorical columns are irrelavant because we use them in the conditional vector for the GAN.
        # Hence, missing values are not a problem, because using training-by-sampling method, the proportions
        # of all categories are equalised
        irrelevant_missing_columns = set(self.categorical_columns)
        relevant_missing_columns = list(all_columns - irrelevant_missing_columns)

        # Replacing missing values with a dummy value: -9999999
        # For continuous columns missing values, it treats `missing` value as a categorical mode and treats the continuous column as a mixed_column
        for i in relevant_missing_columns:

            # Missing values in mixed columns
            if i in list(self.mixed_columns.keys()):
                if "empty" in list(self.data[i].values):
                    self.data[i] = self.data[i].apply(lambda x: -9999999 if x == "empty" else x)

                    # Adding another categorical mode
                    self.mixed_columns[i].append(-9999999)

            # Missing values in continuous columns
            else:
                if "empty" in list(self.data[i].values):
                    self.data[i] = self.data[i].apply(lambda x: -9999999 if x == "empty" else x)

                    # Adding a mixed column with a single categorical mode
                    self.mixed_columns[i] = [-9999999]

    """
  Replacing log_columns as: 
  log(x) if min_value of x > 0
  log(x-min_value+epsilon) if min_value of x < 0  
  """

    def _process_log_columns(self):

        for log_column in self.log_columns:
            valid_indices = []

            # valid indices are indices of non-missing values
            for idx, val in enumerate(self.data[log_column].values):
                if val != -9999999:
                    valid_indices.append(idx)

            eps = 1

            # Lowest valid value
            min_val = np.min(self.data[log_column].iloc[valid_indices].values)
            self.lower_bounds[log_column] = min_val

            # Transformation of log-columns
            if min_val > 0:
                self.data[log_column] = self.data[log_column].apply(lambda x: np.log(x) if x != -9999999 else -9999999)
            elif min_val == 0:
                self.data[log_column] = self.data[log_column].apply(lambda x: np.log(x + eps) if x != -9999999 else -9999999)
            else:
                self.data[log_column] = self.data[log_column].apply(lambda x: np.log(x - min_val + eps) if x != -9999999 else -9999999)

    def _encode_categorical_columns(self):

        # Encoding categorical_columns as numerical classes & storing the corresponding label_encoders in a list for inverse transformation
        for column_index, column in enumerate(self.data.columns):
            if column in self.categorical_columns:

                # Encoding as numerical classes
                label_encoder = preprocessing.LabelEncoder()
                self.data[column] = self.data[column].astype(str)
                label_encoder.fit(self.data[column])
                transformed_column = label_encoder.transform(self.data[column])
                self.data[column] = transformed_column

                # Saving the label_encoder
                current_label_encoder_dict = dict()
                current_label_encoder_dict["column"] = column
                current_label_encoder_dict["label_encoder"] = label_encoder
                self.label_encoder_list.append(current_label_encoder_dict)

                # Storing categorical_column by index
                self.column_types["categorical"].append(column_index)

            elif column in self.mixed_columns:
                # Storing mixed_column by index
                self.column_types["mixed"][column_index] = self.mixed_columns[column]

    """
  Decoding categorical columns
  """

    def _decode_categorical_columns(self):
        # Inverse transforming categorical_columns to classes from numerical values
        for i in range(len(self.label_encoder_list)):
            encoder = self.label_encoder_list[i]["label_encoder"]
            self.inverse_transformed_data[self.label_encoder_list[i]["column"]] = self.inverse_transformed_data[
                self.label_encoder_list[i]["column"]
            ].astype(int)
            self.inverse_transformed_data[self.label_encoder_list[i]["column"]] = encoder.inverse_transform(
                self.inverse_transformed_data[self.label_encoder_list[i]["column"]]
            )

    """
  Exponentiate log-columns
  """

    def _decode_log_columns(self, eps=1):
        for i in self.inverse_transformed_data:
            if i in self.log_columns:
                lower_bound = self.lower_bounds[i]
                if lower_bound > 0:
                    self.inverse_transformed_data = self.inverse_transformed_data[i].apply(lambda x: np.exp(x))
                elif lower_bound == 0:
                    self.inverse_transformed_data[i] = self.inverse_transformed_data[i].apply(
                        lambda x: np.ceil(np.exp(x) - eps) if (np.exp(x) - eps) < 0 else (np.exp(x) - eps)
                    )
                else:
                    self.inverse_transformed_data[i] = self.inverse_transformed_data[i].apply(lambda x: np.exp(x) - eps + lower_bound)

    """
  Rounding off values to nearest integer for integer columns
  """

    def _convert_to_integer(self):
        for column in self.integer_columns:
            self.inverse_transformed_data[column] = self.inverse_transformed_data[column].astype(float)
            print(self.inverse_transformed_data[column].dtype)
            self.inverse_transformed_data[column] = np.round(self.inverse_transformed_data[column].values)
            self.inverse_transformed_data[column] = self.inverse_transformed_data[column].astype(int)

    """
  Method to inverse-transform the generated data so that it resembles actual data:
  1. Converts categorical_cols from numerical values to original classes
  2. Exponentiates log_columns 
  3. Rounds of values to nearest integer for integer_columns
  """

    def inverse_prep(self, data, eps=1):

        self.inverse_transformed_data = pd.DataFrame(data, columns=self.data.columns)

        # Inverse transforming categorical_columns to classes from numerical values
        self._decode_categorical_columns()

        # Exponentiating log_columns to original values
        if self.log_columns:
            self._decode_log_columns(eps)

        # Rounding off values to nearest integers for integer_columns
        if self.integer_columns:
            self._convert_to_integer()

        # Replacing dummy value(s) with NaN
        self.inverse_transformed_data.replace(-9999999, np.nan, inplace=True)
        self.inverse_transformed_data.replace("empty", np.nan, inplace=True)

        return self.inverse_transformed_data


class DataEncoder:
    """
    Description of class members:
    1. self.meta : list[dict()]: stores metadata about all the columns in the train_data
    2. self.n_clusters: integer: maximum number of clusters to be used in the Bayesian Gaussian Mixture technique
    3. self.eps: float: minimum weight of a gaussian in the bayesian gaussian mixture
    4. self.categorical_columns: list: list of indices of categorical columns
    5. self.mixed_columns: dict: keys=name of the mixed column, value=list of the categorical modes of the column
    6. self.train_data: pd.DataFrame: stores the training data
    7. self.ordering:
    8. self.output_info: list(tuple): 1st element of the tuple is an integer, 2nd element is one of 'softmax'/'tanh' indicating the number of elements along with their corresponding activations
    9. self.output_dim: integer: size of output encoding of a data row
    10. self.components: list[list[bool]]: each element is a boolean list indicating whether the i'th mode is essential or not.
    11. self.mixed_col_continuous_filter:list[list[bool]]: each element is a boolean list indicating whether the data point in the mixed column is a part of the continuous part
    """

    """
  Constructor to initialize the class variables
  """

    def __init__(self, train_data: pd.DataFrame, categorical_list=[], mixed_dict={}, n_clusters=10, eps=0.005):
        self.meta = None
        self.n_clusters = n_clusters
        self.eps = eps
        self.train_data = train_data
        self.categorical_columns = categorical_list
        self.mixed_columns = mixed_dict

    """
  Function to get metadata about the different types of columns in the dataset.
  1. categorical columns - sends the various (numerical) categories
  2. mixed columns - sends min and max values for the column, along with the discrete modes of mixed col
  3. continuous columns - sends min and max values 
  """

    def get_metadata(self):

        meta = []
        for index in range(self.train_data.shape[1]):

            column = self.train_data.iloc[:, index]
            if index in self.categorical_columns:
                mapper = column.value_counts().index.tolist()
                meta.append({"name": index, "type": "categorical", "size": len(mapper), "i2s": mapper})
            elif index in self.mixed_columns.keys():
                meta.append({"name": index, "type": "mixed", "min": column.min(), "max": column.max(), "modes": self.mixed_columns[index]})
            else:
                meta.append(
                    {
                        "name": index,
                        "type": "continuous",
                        "min": column.min(),
                        "max": column.max(),
                    }
                )
        return meta

    """
  Function to fit a bayesian gaussian mixture model to the given data.
  """

    def _fit_gaussian_mixture(self, data: np.array):
        gaussian_mixture = BayesianGaussianMixture(
            n_components=self.n_clusters,
            weight_concentration_prior=0.001,
            weight_concentration_prior_type="dirichlet_process",
            max_iter=100,
            n_init=1,
            random_state=42,
        )
        gaussian_mixture.fit(data.reshape([-1, 1]))
        return gaussian_mixture

    """
  Function to find a mask indicating whether or not the ith mode in the gaussian mixture is essential.
  """

    def _find_essential_modes(self, data: np.array, gaussian_mixture: BayesianGaussianMixture):
        mask = gaussian_mixture.weights_ > self.eps
        modes = pd.Series(gaussian_mixture.predict(data.reshape([-1, 1]))).value_counts().keys()
        component_mask = []
        for i in range(self.n_clusters):
            if (i in modes) and mask[i]:
                component_mask.append(True)
            else:
                component_mask.append(False)
        return component_mask

    """
  Function to fit gaussian mixtures to the training data
  """

    def fit(self):

        data = self.train_data.values
        self.meta = self.get_metadata()

        model = []
        self.output_info = []
        self.output_dim = 0
        self.components = []

        self.ordering = []
        self.mixed_col_continuous_filter = []

        for index, col_metadata in enumerate(self.meta):
            if col_metadata["type"] == "continuous":

                # fitting a bayesian gaussian mixture model to the data
                continuous_mixture = self._fit_gaussian_mixture(data[:, index])
                model.append(continuous_mixture)

                # identifying essential gaussians from the mixture (essential -> gaussians with a substantial component in the mixture)
                mode_mask = self._find_essential_modes(data[:, index], continuous_mixture)
                self.components.append(mode_mask)

                # output format and information
                self.output_info += [(1, "tanh"), (np.sum(mode_mask), "softmax")]
                self.output_dim += 1 + np.sum(mode_mask)

            elif col_metadata["type"] == "mixed":

                # gaussian mixture of the entire column
                full_mixture = self._fit_gaussian_mixture(data[:, index])

                continuous_filter = []
                for element in data[:, index]:

                    # separating the continuous part of the mixed column - col_metadata['modes'] is a list of the categorical modes
                    if element not in col_metadata["modes"]:
                        continuous_filter.append(True)
                    else:
                        continuous_filter.append(False)
                self.mixed_col_continuous_filter.append(continuous_filter)

                # continuous data in the mixed column
                continuous_data = data[:, index][continuous_filter]

                # fitting a gaussian mixture to the continuous data
                continuous_mixture = self._fit_gaussian_mixture(continuous_data)
                model.append((full_mixture, continuous_mixture))

                # identifying essential gaussians
                mode_mask = self._find_essential_modes(continuous_data, continuous_mixture)
                self.components.append(mode_mask)

                # output format and information
                self.output_info += [(1, "tanh"), (np.sum(mode_mask) + len(col_metadata["modes"]), "softmax")]
                self.output_dim += 1 + np.sum(mode_mask) + len(col_metadata["modes"])

            else:
                # Categorical columns
                model.append(None)
                self.components.append(None)
                self.output_info += [(col_metadata["size"], "softmax")]
                self.output_dim += col_metadata["size"]
        self.model = model

    def transform(self, data, ispositive=False, positive_list=None):
        # Transformed values after encoding
        values = []
        mixed_counter = 0

        for id_, info in enumerate(self.meta):
            # Current columns being encoded
            current = data[:, id_]

            if info["type"] == "continuous":
                current = current.reshape([-1, 1])

                # Mean of the various Gaussians in the mixture
                means = self.model[id_].means_.reshape((1, self.n_clusters))
                # Standard deviation of the various Gaussians in the mixture
                stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))

                features = np.empty(shape=(len(current), self.n_clusters))
                if ispositive == True:
                    if id_ in positive_list:
                        features = np.abs(current - means) / (4 * stds)
                else:
                    features = (current - means) / (4 * stds)

                # Probability of each data-point belonging to each Gaussian cluster
                probs = self.model[id_].predict_proba(current.reshape([-1, 1]))
                n_opts = sum(self.components[id_])

                # Valid clusters in the gaussian mixture and their corresponding probabilities
                features = features[:, self.components[id_]]
                probs = probs[:, self.components[id_]]

                opt_sel = np.zeros(len(data), dtype="int")
                for i in range(len(data)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)

                    # Encoding value in accordance with the probability of being associated to each cluster in the gaussian mixture
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

                idx = np.arange((len(features)))

                # Assigning features with encodings
                features = features[idx, opt_sel].reshape([-1, 1])
                features = np.clip(features, -0.99, 0.99)

                # One-hot encoding to represent the mode in the Gaussian mixture
                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1

                re_ordered_phot = np.zeros_like(probs_onehot)

                col_sums = probs_onehot.sum(axis=0)

                n = probs_onehot.shape[1]
                largest_indices = np.argsort(-1 * col_sums)[:n]
                self.ordering.append(largest_indices)
                for id, val in enumerate(largest_indices):
                    re_ordered_phot[:, id] = probs_onehot[:, val]

                values += [features, re_ordered_phot]

            elif info["type"] == "mixed":

                means_0 = self.model[id_][0].means_.reshape([-1])
                stds_0 = np.sqrt(self.model[id_][0].covariances_).reshape([-1])

                zero_std_list = []
                means_needed = []
                stds_needed = []

                for mode in info["modes"]:
                    if mode != -9999999:
                        dist = []
                        for idx, val in enumerate(list(means_0.flatten())):
                            dist.append(abs(mode - val))
                        index_min = np.argmin(np.array(dist))
                        zero_std_list.append(index_min)
                    else:
                        continue

                for idx in zero_std_list:
                    means_needed.append(means_0[idx])
                    stds_needed.append(stds_0[idx])

                mode_vals = []

                for i, j, k in zip(info["modes"], means_needed, stds_needed):
                    this_val = np.abs(i - j) / (4 * k)
                    mode_vals.append(this_val)

                if -9999999 in info["modes"]:
                    mode_vals.append(0)

                current = current.reshape([-1, 1])
                filter_arr = self.mixed_col_continuous_filter[mixed_counter]
                current = current[filter_arr]

                means = self.model[id_][1].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_][1].covariances_).reshape((1, self.n_clusters))
                features = np.empty(shape=(len(current), self.n_clusters))
                if ispositive == True:
                    if id_ in positive_list:
                        features = np.abs(current - means) / (4 * stds)
                else:
                    features = (current - means) / (4 * stds)

                probs = self.model[id_][1].predict_proba(current.reshape([-1, 1]))

                n_opts = sum(self.components[id_])  # 8
                features = features[:, self.components[id_]]
                probs = probs[:, self.components[id_]]

                opt_sel = np.zeros(len(current), dtype="int")
                for i in range(len(current)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)
                idx = np.arange((len(features)))
                features = features[idx, opt_sel].reshape([-1, 1])
                features = np.clip(features, -0.99, 0.99)
                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1
                extra_bits = np.zeros([len(current), len(info["modes"])])
                temp_probs_onehot = np.concatenate([extra_bits, probs_onehot], axis=1)
                final = np.zeros([len(data), 1 + probs_onehot.shape[1] + len(info["modes"])])
                features_curser = 0
                for idx, val in enumerate(data[:, id_]):
                    if val in info["modes"]:
                        category_ = list(map(info["modes"].index, [val]))[0]
                        final[idx, 0] = mode_vals[category_]
                        final[idx, (category_ + 1)] = 1

                    else:
                        final[idx, 0] = features[features_curser]
                        final[idx, (1 + len(info["modes"])) :] = temp_probs_onehot[features_curser][len(info["modes"]) :]
                        features_curser = features_curser + 1

                just_onehot = final[:, 1:]
                re_ordered_jhot = np.zeros_like(just_onehot)
                n = just_onehot.shape[1]
                col_sums = just_onehot.sum(axis=0)
                largest_indices = np.argsort(-1 * col_sums)[:n]
                self.ordering.append(largest_indices)
                for id, val in enumerate(largest_indices):
                    re_ordered_jhot[:, id] = just_onehot[:, val]
                final_features = final[:, 0].reshape([-1, 1])
                values += [final_features, re_ordered_jhot]
                mixed_counter = mixed_counter + 1

            else:
                self.ordering.append(None)
                col_t = np.zeros([len(data), info["size"]])
                idx = list(map(info["i2s"].index, current))
                col_t[np.arange(len(data)), idx] = 1
                values.append(col_t)

        return np.concatenate(values, axis=1)

    def inverse_transform(self, data):
        data_t = np.zeros([len(data), len(self.meta)])
        st = 0
        for id_, info in enumerate(self.meta):
            if info["type"] == "continuous":
                u = data[:, st]
                v = data[:, st + 1 : st + 1 + np.sum(self.components[id_])]
                order = self.ordering[id_]
                v_re_ordered = np.zeros_like(v)

                for id, val in enumerate(order):
                    v_re_ordered[:, val] = v[:, id]

                v = v_re_ordered

                u = np.clip(u, -1, 1)
                v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                v_t[:, self.components[id_]] = v
                v = v_t
                st += 1 + np.sum(self.components[id_])
                means = self.model[id_].means_.reshape([-1])
                stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]
                tmp = u * 4 * std_t + mean_t
                data_t[:, id_] = tmp

            elif info["type"] == "mixed":

                u = data[:, st]
                full_v = data[:, (st + 1) : (st + 1) + len(info["modes"]) + np.sum(self.components[id_])]
                order = self.ordering[id_]
                full_v_re_ordered = np.zeros_like(full_v)

                for id, val in enumerate(order):
                    full_v_re_ordered[:, val] = full_v[:, id]

                full_v = full_v_re_ordered
                mixed_v = full_v[:, : len(info["modes"])]
                v = full_v[:, -np.sum(self.components[id_]) :]

                u = np.clip(u, -1, 1)
                v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                v_t[:, self.components[id_]] = v
                v = np.concatenate([mixed_v, v_t], axis=1)

                st += 1 + np.sum(self.components[id_]) + len(info["modes"])
                means = self.model[id_][1].means_.reshape([-1])
                stds = np.sqrt(self.model[id_][1].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)

                result = np.zeros_like(u)

                for idx in range(len(data)):
                    if p_argmax[idx] < len(info["modes"]):
                        argmax_value = p_argmax[idx]
                        result[idx] = float(list(map(info["modes"].__getitem__, [argmax_value]))[0])
                    else:
                        std_t = stds[(p_argmax[idx] - len(info["modes"]))]
                        mean_t = means[(p_argmax[idx] - len(info["modes"]))]
                        result[idx] = u[idx] * 4 * std_t + mean_t

                data_t[:, id_] = result

            else:
                current = data[:, st : st + info["size"]]
                st += info["size"]
                idx = np.argmax(current, axis=1)
                data_t[:, id_] = list(map(info["i2s"].__getitem__, idx))

        return data_t


class TransformBatch:
    """
    The GAN uses Conv2D and DeConv2D layers which use 2D inputs of length and width close to each other.
    The TransformBatch class pads the batch with 0's to the appropriate width.
    """

    def __init__(self, side):
        self.height = side

    def transform(self, data):
        if self.height * self.height > len(data[0]):
            padding = torch.zeros((len(data), self.height * self.height - len(data[0]))).to(data.device)
            data = torch.cat([data, padding], axis=1)
        return data.view(-1, 1, self.height, self.height)

    def inverse_transform(self, data):
        data = data.view(-1, self.height * self.height)
        return data


class RealDataSampler:
    """
    Class to implement training-by-sampling and sample real-data
    Description of class members:
    1. self.data: np.array: Encoded training data
    2. self.model: list: The i'th value in the list is list of list of indices of data-points, where the j'th list in the i'th list corresponds to the j'th mode of the i'th column
    3. self.length: integer: length of training data
    """

    def __init__(self, data, output_info):
        super(RealDataSampler, self).__init__()
        self.data = data
        self.model = []
        self.length = len(data)

        # Store the indices of data-rows belonging to each mode of the current column
        self._get_category_rows(output_info)

    """
  For each column (having n modes), the function stores the indices of the rows belonging to the i'th mode in the column
  """

    def _get_category_rows(self, output_info):
        # Start of the current column
        start = 0

        for info in output_info:
            if info[1] == "tanh":
                start += info[0]
                continue
            elif info[1] == "softmax":
                end = start + info[0]

                temp = []
                for j in range(info[0]):
                    # Get all row_indices for the current value of the category
                    temp.append(np.nonzero(self.data[:, start + j])[0])
                self.model.append(temp)
                start = end

    """
  For the given mode of the given column, this function samples 'num_samples' number of rows from the training data 
  """

    def sample(self, num_samples, column, mode):
        # If column is not specified, return 'num_samples' number of random columns
        if column is None:
            idx = np.random.choice(np.arange(self.length), num_samples)
            return self.data[idx]

        idx = []
        for col, mode_value in zip(column, mode):
            # Randomly choose data-points corresponding to the given mode of the given class
            idx.append(np.random.choice(self.model[col][mode_value]))
        return self.data[idx]


class ConditionalVectorGenerator:
    """
    Class to generate the conditional vector to be fed to the GAN for conditional training
    Description of class members:
    1. self.model: list: For each column, it stores a list corresponding to the mode the data-point corresponds to
    2. self.interval:
    3. self.col_count:
    4. self.mode_count:
    5. self.p:
    6. self.p_sampling
    7. self.data:
    """

    def __init__(self, data, output_info):

        self.data = data
        self.model = []
        start = 0
        counter = 0
        for info in output_info:

            if info[1] == "tanh":
                start += info[0]
                continue

            elif info[1] == "softmax":
                end = start + info[0]
                counter += 1
                # Since maximum value is 1, the argmax returns the mode to which the data-point corresponds
                self.model.append(np.argmax(data[:, start:end], axis=-1))
                start = end

        self.get_probabilities(counter, output_info)

    """
  Function to return the maximum number of modes in any column of the training data
  """

    def maximum_modes(self, output_info):
        max_interval = 0
        for item in output_info:
            max_interval = max(max_interval, item[0])
        return max_interval

    """
  Function to calculate probabilities of occurence of each mode for all columns
  """

    def get_probabilities(self, counter, output_info):
        self.interval = []
        self.col_count = 0
        self.mode_count = 0
        start = 0
        self.p = np.zeros((counter, self.maximum_modes(output_info)))
        self.p_sampling = []
        for item in output_info:
            if item[1] == "tanh":
                start += item[0]
                continue
            elif item[1] == "softmax":
                end = start + item[0]

                # Calculate number/frequency of occurences of each mode
                temp = np.sum(self.data[:, start:end], axis=0)
                temp_sampling = np.sum(self.data[:, start:end], axis=0)

                # Finding probabilities from frequencies
                temp_sampling = temp_sampling / np.sum(temp_sampling)
                self.p_sampling.append(temp_sampling)

                # Taking log probabilities
                temp = np.log(temp + 1)
                temp = temp / np.sum(temp)
                self.p[self.col_count, : item[0]] = temp

                # Interval of each mode -> (start-index, number of modes)
                self.interval.append((self.mode_count, item[0]))
                self.mode_count += item[0]
                self.col_count += 1
                start = end

        self.interval = np.asarray(self.interval)

    """
  Function to randomly choose a mode for each column given the probability of each mode --> while sampling data
  """

    def random_choice_prob_index_sampling(self, probs, col_idx):
        option_list = []
        for i in col_idx:
            pp = probs[i]
            option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))

        return np.array(option_list).reshape(col_idx.shape)

    """
  Function to randomly choose a mode for each column given the probability of each mode --> during training
  """

    def random_choice_prob_index(self, a, axis=1):
        r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
        return (a.cumsum(axis=axis) > r).argmax(axis=axis)

    """
  Function to generate conditional vector during training
  """

    def sample_train(self, batch):
        if self.col_count == 0:
            return None
        batch = batch

        # Chooses a random column for the conditional vector for each data-point in the batch
        chosen_column = np.random.choice(np.arange(self.col_count), batch)
        # Conditional vector to choose a particular mode
        vec = np.zeros((batch, self.mode_count), dtype="float32")
        # Binary mask corresponding to the column whose mode has been chosen
        mask = np.zeros((batch, self.col_count), dtype="float32")
        mask[np.arange(batch), chosen_column] = 1

        # Choosing which mode to sample in the conditional vector
        chosen_mode = self.random_choice_prob_index(self.p[chosen_column])
        for i in np.arange(batch):
            vec[i, self.interval[chosen_column[i], 0] + chosen_mode[i]] = 1

        # vec - shape(batch_size, mode_count) -> 1 conditional vector (of size #modes) for each row in the batch
        # mask - shape(batch, column_count) -> 1 binary mask (of size #columns) for each row in the batch
        # chosen_column - shape(batch_size,) -> index of the chosen column for each row in the batch
        # chosen_modes - shape(batch_size,) -> index of the chosen mode of the chosen column for each row in the batch
        return vec, mask, chosen_column, chosen_mode

    """
  Function to generate conditional vector while generating data
  """

    def sample(self, batch):
        if self.col_count == 0:
            return None
        batch = batch
        # Choosing a column to condition the GAN with
        chosen_column = np.random.choice(np.arange(self.col_count), batch)
        # Choosing the mode of the chosen column
        chosen_mode = self.random_choice_prob_index_sampling(self.p_sampling, chosen_column)
        vec = np.zeros((batch, self.mode_count), dtype="float32")
        for i in np.arange(batch):
            vec[i, self.interval[chosen_column[i], 0] + chosen_mode[i]] = 1

        return vec


class Generator(Module):
    """
    Class to generate tabular data from random noise using Transponse-CNNs
    """

    def __init__(self, side, random_dims, num_channels):
        super(Generator, self).__init__()
        self.side = side
        self.layers = self.determine_layers(side, random_dims, num_channels)
        self.model = Sequential(*self.layers)

    def forward(self, input):
        return self.model(input)

    """
  Function to determine layers of the generator
  """

    def determine_layers(self, side, random_dims, num_channels):
        assert side >= 4 and side <= 32

        # Model dimensions and architecture
        layer_dims = [(1, side), (num_channels, side // 2)]

        while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
            layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

        # Adding ConvTranspose2D layers as part of the model
        layers = [ConvTranspose2d(random_dims, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)]

        for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
            layers += [BatchNorm2d(prev[0]), ReLU(True), ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)]
        return layers


class Discriminator(Module):
    """
    Class to discriminate between real and fake data using CNNs
    """

    def __init__(self, side, num_channels):
        super(Discriminator, self).__init__()
        self.side = side
        self.layers = self.determine_layers(side, num_channels)
        info = len(self.layers) - 2
        self.model = Sequential(*self.layers)
        self.model_info = Sequential(*self.layers[:info])

    def forward(self, input):
        return (self.model(input)), self.model_info(input)

    def determine_layers(self, side, num_channels):
        assert side >= 4 and side <= 32

        # Model dimensions and architecture
        layer_dims = [(1, side), (num_channels, side // 2)]

        while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
            layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

        # Adding Conv2D layers in the discriminator architecture
        layers = []
        for prev, curr in zip(layer_dims, layer_dims[1:]):
            layers += [Conv2d(prev[0], curr[0], 4, 2, 1, bias=False), BatchNorm2d(curr[0]), LeakyReLU(0.2, inplace=True)]
        layers += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0), Sigmoid()]

        return layers


class Classifier(Module):
    """
    Auxillary classifier as per the CTAB-GAN paper
    """

    def __init__(self, input_dim, layer_dims, target_bounds):
        super(Classifier, self).__init__()
        target_label_length = target_bounds[1] - target_bounds[0]
        input_dim = input_dim - (target_label_length)
        seq = []
        self.bounds = target_bounds

        # Classifier model layers
        for item in list(layer_dims):
            seq += [Linear(input_dim, item), LeakyReLU(0.2), Dropout(0.5)]
            input_dim = item

        # Final layer in the classifier
        if (target_label_length) == 1:
            seq += [Linear(input_dim, 1)]
        elif (target_label_length) == 2:
            seq += [Linear(input_dim, 1), Sigmoid()]
        else:
            seq += [Linear(input_dim, target_label_length)]
        self.model = Sequential(*seq)

    def forward(self, input):
        # Target label
        label = None

        # Deciphering label for the classifier
        if (self.bounds[1] - self.bounds[0]) == 1:
            label = input[:, self.bounds[0] : self.bounds[1]]
        else:
            label = torch.argmax(input[:, self.bounds[0] : self.bounds[1]], axis=-1)

        # Removing the target column from the input features
        new_input = torch.cat((input[:, : self.bounds[0]], input[:, self.bounds[1] :]), 1)
        if ((self.bounds[1] - self.bounds[0]) == 2) | ((self.bounds[1] - self.bounds[0]) == 1):
            return self.model(new_input).view(-1), label
        else:
            return self.model(new_input), label


class Utils:
    """
    Class that offers utility functions
    """

    """
    Function that applies the respective activations to the data generated by the generator
    """

    @staticmethod
    def apply_activations(data, output_info):
        data_activated = []
        start = 0
        for item in output_info:
            if item[1] == "tanh":
                end = start + item[0]
                data_activated.append(torch.tanh(data[:, start:end]))
                start = end
            elif item[1] == "softmax":
                end = start + item[0]
                data_activated.append(F.gumbel_softmax(data[:, start:end], tau=0.2))
                start = end
        return torch.cat(data_activated, dim=1)

    """
    Conditional GAN loss
    """

    @staticmethod
    def conditional_loss(data, output_info, c, m):
        loss = []
        st = 0
        st_c = 0
        for item in output_info:
            if item[1] == "tanh":
                st += item[0]
                continue

            elif item[1] == "softmax":
                ed = st + item[0]
                ed_c = st_c + item[0]
                tmp = F.cross_entropy(data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction="none")
                loss.append(tmp)
                st = ed
                st_c = ed_c

        loss = torch.stack(loss, dim=1)
        return (loss * m).sum() / data.size()[0]


class DataSynthesizer:
    """
    Class to train the GAN model and generate synthetic data samples from it.
    Description of class members:
    1. self.random_dims: integer: dimensions of the noise vector input to the GAN
    2. self.classifier_dims: list[integer]: classifier architecture -> dimensions of the layers of the classifier
    3. self.num_channels:
    4. self.dside:
    5. self.gside:
    6. self.l2scale:
    7. self.batch_size: integer: batch_size used while training
    8. self.epochs: integer:
    9. self.device: "cuda:0"/"cpu" :device on which to train the model
    """

    """
    Constructor to initialize class members
    """

    def __init__(self, classifier_dims=(256, 256, 256, 256), random_dims=100, num_channels=64, weight_decay=1e-5, batch_size=500, epochs=1):

        self.random_dims = random_dims
        self.classifier_dims = classifier_dims
        self.num_channels = num_channels
        self.dside = None
        self.gside = None
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """
    Utility function to determine the input dimensions to the Generator and Discriminator (which use Conv2D Layers)
    Maximum Encoding - 
    """

    def _find_input_side_length(self, input_dims):
        side = 0
        sides = [4, 8, 16, 24, 32]
        for i in sides:
            if i * i >= input_dims:
                side = i
                break
        return side

    """
  Utility function to find the bounds (in the encoded training-data) of the target-column for the classifier 
  """

    def target_column_bounds(self, target_col_index, output_info):
        start = 0
        count = 0
        total_count = 0
        for item in output_info:
            if count == target_col_index:
                break
            if item[1] == "tanh":
                start += item[0]
            elif item[1] == "softmax":
                start += item[0]
                count += 1
            total_count += 1
        end = start + output_info[total_count][0]
        return (start, end)

    """
  Initialize model weights
  """

    def weights_init(self, m):
        classname = m.__class__.__name__

        if classname.find("Conv") != -1:
            init.normal_(m.weight.data, 0.0, 0.02)

        elif classname.find("BatchNorm") != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0)

    """
  Function to fit the CTAB-GAN model
  """

    def fit(
        self,
        train_data: pd.DataFrame,
        categorical_cols=[],
        mixed_cols={},
        problem_type={},
        save_weights=True,
        save_frequency=50,
        save_path="./weigths",
        load_weights=False,
        pre_trained_paths={},
    ):

        # Setting the problem type - Classification/Regression and the target-column
        problem = None
        target_column_index = None
        if problem_type:
            problem = list(problem_type.keys())[0]
            target_column_index = train_data.columns.get_loc(problem_type[problem])

        # DataEncoder class to receive the preprocessed data (output from the DataPreprocessing class)
        self.encoder = DataEncoder(train_data, categorical_list=categorical_cols, mixed_dict=mixed_cols)
        self.encoder.fit()
        # Encoding the training data
        train_data = self.encoder.transform(train_data.values)

        output_info = self.encoder.output_info
        # data_sampler to sample real data using training-by-sampling
        data_sampler = RealDataSampler(train_data, output_info)

        encoded_data_dims = self.encoder.output_dim

        self.conditional_vector_generator = ConditionalVectorGenerator(train_data, output_info)
        # Finding dimensions of the input data to the generator and discriminator
        self.dside = self._find_input_side_length(encoded_data_dims + self.conditional_vector_generator.mode_count)
        self.gside = self._find_input_side_length(encoded_data_dims)

        # Defining the model

        self.generator = Generator(self.gside, self.random_dims + self.conditional_vector_generator.mode_count, self.num_channels).to(self.device)
        discriminator = Discriminator(self.dside, self.num_channels).to(self.device)

        # Optimizers
        optimizer_params = dict(lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.weight_decay)
        optimizer_generator = Adam(self.generator.parameters(), **optimizer_params)
        optimizer_discriminator = Adam(discriminator.parameters(), **optimizer_params)

        # Classifier
        bounds = None
        classifier = None
        optimizer_classifier = None
        if target_column_index != None:
            bounds = self.target_column_bounds(target_column_index, output_info)
            classifier = Classifier(encoded_data_dims, self.classifier_dims, bounds).to(self.device)
            optimizer_classifier = optim.Adam(classifier.parameters(), **optimizer_params)

        # Initializing weights
        print(self.gside, self.random_dims + self.conditional_vector_generator.mode_count, self.num_channels)
        if load_weights:
            self.generator.load_state_dict(torch.load(pre_trained_paths["generator"]))
            # discriminator.load_state_dict(torch.load(pre_trained_paths["discriminator"]))
            # classifier.load_state_dict(torch.load(pre_trained_paths["classifier"]))
            print("Loaded pre_trained models!")
        else:
            self.generator.apply(self.weights_init)
            discriminator.apply(self.weights_init)

        self.gen_transform = TransformBatch(self.gside)
        self.disc_transform = TransformBatch(self.dside)

        steps_per_epoch = max(1, len(train_data) // self.batch_size)

        # Train the model
        for i in tqdm(range(self.epochs)):
            for _ in range(steps_per_epoch):
                # Random noise to be fed to the generator
                noise = torch.randn(self.batch_size, self.random_dims, device=self.device)

                # Generating conditional vector
                conditional_vector = self.conditional_vector_generator.sample_train(self.batch_size)
                vec, mask, chosen_column, chosen_mode = conditional_vector

                # Concatenating noise with conditional vector
                vec = torch.from_numpy(vec).to(self.device)
                mask = torch.from_numpy(mask).to(self.device)
                noise = torch.cat([noise, vec], dim=1)
                noise = noise.view(self.batch_size, self.random_dims + self.conditional_vector_generator.mode_count, 1, 1)

                # Randomly shuffling the input
                permute = np.arange(self.batch_size)
                np.random.shuffle(permute)

                # Sampling real data as per the conditional vector
                real_data = data_sampler.sample(self.batch_size, chosen_column[permute], chosen_mode[permute])
                vec_permuted = vec[permute]

                real_data = torch.from_numpy(real_data.astype("float32")).to(self.device)

                # Generating synthetic data
                fake_data = self.generator(noise)
                fake_data = self.gen_transform.inverse_transform(fake_data)
                fake_data = Utils.apply_activations(fake_data, output_info)

                conditional_fake = torch.cat([fake_data, vec], dim=1)
                conditional_real = torch.cat([real_data, vec_permuted], dim=1)
                real_data_transformed = self.disc_transform.transform(conditional_real)
                fake_data_transformed = self.disc_transform.transform(conditional_fake)

                # Training the discriminator
                optimizer_discriminator.zero_grad()
                y_real, _ = discriminator(real_data_transformed)
                y_fake, _ = discriminator(fake_data_transformed)
                loss_d = -(torch.log(y_real + 1e-4).mean()) - (torch.log(1.0 - y_fake + 1e-4).mean())
                loss_d.backward()
                optimizer_discriminator.step()

                # Training the generator
                noise = torch.randn(self.batch_size, self.random_dims, device=self.device)
                conditional_vector = self.conditional_vector_generator.sample_train(self.batch_size)
                vec, mask, chosen_column, chosen_mode = conditional_vector

                vec = torch.from_numpy(vec).to(self.device)
                mask = torch.from_numpy(mask).to(self.device)
                noise = torch.cat([noise, vec], dim=1)
                noise = noise.view(self.batch_size, self.random_dims + self.conditional_vector_generator.mode_count, 1, 1)

                optimizer_generator.zero_grad()
                fake_data = self.generator(noise)
                fake_data_transform = self.gen_transform.inverse_transform(fake_data)
                fake_data = Utils.apply_activations(fake_data_transform, output_info)

                conditional_fake = torch.cat([fake_data, vec], dim=1)
                fake_data_transformed = self.disc_transform.transform(conditional_fake)
                y_fake, info_fake = discriminator(fake_data_transformed)

                # Calculating loss function
                cross_entropy = Utils.conditional_loss(fake_data_transform, output_info, vec, mask)
                _, info_real = discriminator(real_data_transformed)

                g = -(torch.log(y_fake + 1e-4).mean()) + cross_entropy

                # Backpropagating through the generator loss
                g.backward(retain_graph=True)
                loss_mean = torch.norm(
                    torch.mean(info_fake.view(self.batch_size, -1), dim=0) - torch.mean(info_real.view(self.batch_size, -1), dim=0), 1
                )
                loss_std = torch.norm(
                    torch.std(info_fake.view(self.batch_size, -1), dim=0) - torch.std(info_real.view(self.batch_size, -1), dim=0), 1
                )
                loss_info = loss_mean + loss_std
                loss_info.backward()

                optimizer_generator.step()

                # Classifying the generated data to ensure semantic integrity
                if problem:
                    fake_data = self.generator(noise)
                    fake_data = self.gen_transform.inverse_transform(fake_data)
                    fake_data = Utils.apply_activations(fake_data, output_info)

                    real_input, real_label = classifier(real_data)
                    fake_input, fake_label = classifier(fake_data)

                    loss = CrossEntropyLoss()

                    if (bounds[1] - bounds[0]) == 1:
                        loss = SmoothL1Loss()
                        real_label = real_label.type_as(real_input)
                        fake_label = fake_label.type_as(fake_input)
                        real_label = torch.reshape(real_label, real_input.size())
                        fake_label = torch.reshape(fake_label, fake_input.size())

                    elif bounds[1] - bounds[0] == 2:
                        loss = BCELoss()
                        real_label = real_label.type_as(real_input)
                        fake_label = fake_label.type_as(fake_input)

                    # Calculating classifier and generator loss
                    classifier_loss = loss(real_input, real_label)
                    generator_loss = loss(fake_input, fake_label)

                    # Backpropagation through the classifier
                    optimizer_generator.zero_grad()
                    generator_loss.backward()
                    optimizer_generator.step()

                    optimizer_classifier.zero_grad()
                    classifier_loss.backward()
                    optimizer_classifier.step()

            if save_weights and (i + 1) % save_frequency == 0:
                torch.save(self.generator.state_dict(), f"{save_path}/transfer-generator-epoch-{i+1}.h5")
                torch.save(discriminator.state_dict(), f"{save_path}/transfer-discriminator-epoch-{i+1}.h5")
                torch.save(classifier.state_dict(), f"{save_path}/transfer-classifier-epoch-{i+1}.h5")

    """
  Function to use the trained model to generate synthetic data
  """

    def sample(self, n):

        self.generator.eval()
        output_info = self.encoder.output_info
        steps = n // self.batch_size + 1
        data = []

        for i in range(steps):
            # Random noise generated to be fed to the generator
            noise = torch.randn(self.batch_size, self.random_dims, device=self.device)
            # Generating the conditional vector
            conditional_vec = self.conditional_vector_generator.sample(self.batch_size)

            # Generating synthetic encoded data
            conditional_vec = torch.from_numpy(conditional_vec).to(self.device)
            noise = torch.cat([noise, conditional_vec], dim=1)
            noise = noise.view(self.batch_size, self.random_dims + self.conditional_vector_generator.mode_count, 1, 1)
            fake_data = self.generator(noise)

            # Decoding and inverse transforming the generated data
            fake_data = self.gen_transform.inverse_transform(fake_data)
            fake_data = Utils.apply_activations(fake_data, output_info)
            data.append(fake_data.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        result = self.encoder.inverse_transform(data)

        return result[0:n]


class CTABGAN:
    """
    CTAB-GAN Module to train the CTAB-GAN model in an abstract manner
    """

    """
  Constructor to initialize class functions
  """

    def __init__(
        self,
        input_csv_path="./adult.csv",
        categorical_columns=["workclass", "education", "marital-status", "occupation", "relationship", "race", "gender", "native-country", "income"],
        log_columns=[],
        mixed_columns={"capital-loss": [0.0], "capital-gain": [0.0]},
        integer_columns=["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"],
        problem_type={"Classification": "income"},
        epochs=1,
        save_weights=True,
        save_frequency=50,
        save_path="./weights",
        load_weights=False,
        pre_trained_paths={},
    ):

        self.__name__ = "CTABGAN"
        self.synthesizer = DataSynthesizer(epochs=epochs)
        self.raw_data = pd.read_csv(input_csv_path)
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        self.save_weights = save_weights
        self.save_frequency = save_frequency
        self.save_path = save_path
        self.load_weights = load_weights
        self.pre_trained_paths = pre_trained_paths

    """
  Fit the CTAB-GAN model
  """

    def fit(self):

        self.data_prep = DataPreprocessing(
            self.raw_data, self.categorical_columns, self.log_columns, self.mixed_columns, self.integer_columns, self.problem_type
        )
        self.synthesizer.fit(
            train_data=self.data_prep.data,
            categorical_cols=self.data_prep.column_types["categorical"],
            mixed_cols=self.data_prep.column_types["mixed"],
            problem_type=self.problem_type,
            save_weights=self.save_weights,
            save_frequency=self.save_frequency,
            save_path=self.save_path,
            load_weights=self.load_weights,
            pre_trained_paths=self.pre_trained_paths,
        )

    """
  Generate synthetic data using the trained GAN model
  """

    def generate_samples(self, num_samples, synthesizer=None, data_prep=None):
        synthetic_data = None
        if synthesizer is not None:
            synthetic_samples = synthesizer.sample(num_samples)
            synthetic_data = data_prep.inverse_prep(synthetic_samples)
        else:
            synthetic_samples = self.synthesizer.sample(num_samples)
            synthetic_data = self.data_prep.inverse_prep(synthetic_samples)
        return synthetic_data

    def sample_from_pretrained(self, num_samples, output_path="./synthetic-dataset.csv"):
        data_synthesizer = DataSynthesizer(epochs=0)
        data_prep = DataPreprocessing(
            self.raw_data, self.categorical_columns, self.log_columns, self.mixed_columns, self.integer_columns, self.problem_type
        )
        data_synthesizer.fit(
            train_data=data_prep.data,
            categorical_cols=data_prep.column_types["categorical"],
            mixed_cols=data_prep.column_types["mixed"],
            problem_type=self.problem_type,
            save_weights=self.save_weights,
            save_frequency=self.save_frequency,
            load_weights=self.load_weights,
            pre_trained_paths=self.pre_trained_paths,
        )
        data = self.generate_samples(num_samples, synthesizer=data_synthesizer, data_prep=data_prep)
        data.to_csv(output_path, index=False)