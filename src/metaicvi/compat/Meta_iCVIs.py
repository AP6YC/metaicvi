import numpy as np
# from iCVIs import iCVI
from cvi.compat import iCVI
from scipy.stats import spearmanr
from collections import deque
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sktime.utils.data_processing import from_3d_numpy_to_nested


class Meta_iCVI:
    def __init__(self,window_size,icvi_a='iCH', icvi_b='iSIL'):
        self.icvi_a = iCVI(icvi_a)
        self.icvi_b = iCVI(icvi_b)
        self.classification_window_size = None
        self.correlation_history = []
        self.window_size = window_size
        self.icvi_history = {'a':deque(window_size*[0.],window_size), 'b':deque(window_size*[0.],window_size)}
        self.labels = set()
        self.sample_counter = 0

        self.rocket = Rocket(num_kernels=10000,random_state=111)
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        self.is_fit = False

    def reset(self):
        self.icvi_a.__init__()
        self.icvi_b.__init__()
        if self.classification_window_size is None:
            self.correlation_history = []
        else:
            self.correlation_history = deque(self.classification_window_size * [0.], self.classification_window_size)
        self.icvi_history = {'a': deque(self.window_size * [0.], self.window_size), 'b': deque(self.window_size * [0.], self.window_size)}
        self.labels = set()
        self.sample_counter = 0


    def increment(self,sample,label,predict=True,numeric_prediction=False):
        if label not in self.labels and label > len(self.labels):
            raise RuntimeError('Invalid Cluster ordering. Cluster {} is not ordered correctly with respect to exisiting clusters {}'.format(label, self.labels))
        self.sample_counter += 1
        self.labels.add(label)
        self.icvi_a.update(sample,label)
        self.icvi_history['a'].append(self.icvi_a.output)
        self.icvi_b.update(sample,label)
        self.icvi_history['b'].append(self.icvi_b.output)

        if self.sample_counter >= self.window_size:
            a_hist = np.array(self.icvi_history['a'])
            b_hist = np.array(self.icvi_history['b'])
            corr = spearmanr(a_hist, b_hist)
        else:
            corr = 0.
        self.correlation_history.append(corr)

        if predict:
            if not self.is_fit:
                raise RuntimeWarning('Cannot predict partition quality prior to fitting')
            else:
                correlation_sample_array = [np.nan_to_num(np.pad(np.array(self.correlation_history), [self.classification_window_size - len(self.correlation_history), 0]))]
                correlation_sample_array = from_3d_numpy_to_nested(correlation_sample_array.reshape((1, 1, -1)))

                correlation_samples_transformed = self.rocket.transform(correlation_sample_array)

                partition_quality_prediction = self.classifier.predict(correlation_samples_transformed)

                if numeric_prediction:
                    return partition_quality_prediction
                else:
                    return self.reverse_transform_partition_quality_predictions(partition_quality_prediction)


    def validate_partition_quality_labels(self, partition_quality_labels):
        unique_partition_labels = set(np.unique(partition_quality_labels))
        required_partition_labels = {'under', 'over', 'correct'}
        if required_partition_labels != unique_partition_labels:
            if len(unique_partition_labels) < len(required_partition_labels):
                for req_label in required_partition_labels:
                    if req_label not in unique_partition_labels:
                        raise RuntimeWarning('{}-partitioned examples are missing from training set.'.format(req_label))
            for label in unique_partition_labels:
                if label not in required_partition_labels:
                    raise RuntimeError(
                        '{} is not a valid label. Labels must only be {}'.format(label, required_partition_labels))

    def transform_partition_quality_labels(self, partition_quality_labels):
        transformed = [0]*len(partition_quality_labels)
        key = {'under':-1,'over':1,'correct':0}
        for i, label in enumerate(partition_quality_labels):
            transformed[i] = key[label]
        return np.array(transformed)

    def reverse_transform_partition_quality_predictions(self, partition_quality_predictions):
        transformed = [0]*len(partition_quality_predictions)
        key = {-1:'under', 1:'over', 0:'correct'}
        for i, pred in enumerate(partition_quality_predictions):
            transformed[i] = key[pred]
        return np.array(transformed)

    def fit(self,samples_label_pairs, partition_quality_labels):
        self.validate_partition_quality_labels(partition_quality_labels)
        if self.is_fit:
            raise RuntimeWarning('Classifier has already been fit. Fitting again will reset classifier')
        self.reset()
        correlation_samples = []
        for (X,Y),z in zip(samples_label_pairs, partition_quality_labels):
            for x,y in zip(X,Y):
                self.increment(x,y,predict=False)
            correlation_samples.append(np.array(self.correlation_history))
            self.reset()

        self.classification_window_size = max([len(csample) for csample in correlation_samples])
        correlation_sample_array = [np.nan_to_num(np.pad(x, [self.classification_window_size - len(x), 0])) for x in correlation_samples]
        correlation_sample_array = from_3d_numpy_to_nested(correlation_sample_array.reshape((correlation_sample_array.shape[0], 1, -1)))

        self.rocket.fit(correlation_sample_array)
        correlation_samples_transformed = self.rocket.transform(correlation_sample_array)

        partition_quality_labels_transformed = self.transform_partition_quality_labels(partition_quality_labels)
        self.classifier.fit(correlation_samples_transformed, partition_quality_labels_transformed)

        self.reset()
        self.is_fit = True
