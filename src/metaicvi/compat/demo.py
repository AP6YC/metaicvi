import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from Meta_iCVIs import Meta_iCVI
from datasets import Dataset
from sklearn.model_selection import train_test_split
from random import shuffle



dataset_name = 'iris'
examples_to_generate = {
    'under': 6,
    'correct': 6,
    'over': 6
}

def generate_data():
    data = Dataset(dataset_name)
    X = np.vstack(data.data)
    Y = np.array(data.labels)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    sample_label_pairs = []
    partition_quality_labels = []
    cluster_labels = np.unique(Y)
    n_clusters = len(cluster_labels)
    for partition_quality, n_examples in examples_to_generate.items():
        for _ in range(n_examples):
            ridx = np.array(list(range(X_train.shape[0])))
            random.shuffle(ridx)
            X_train = X_train[ridx, :]
            Y_train = Y_train[ridx]

            if partition_quality == 'correct':
                shuffle(cluster_labels)
                shuffle_func = lambda i: cluster_labels[i]
                new_Y_train = np.array(Y_train)
                new_Y_train = np.array(list(map(shuffle_func, new_Y_train)))

            else:
                if partition_quality == 'under':
                    if n_clusters - 1 > 2:
                        k = np.random.randint(2, n_clusters - 1)
                    else:
                        k = 2
                else:
                    k = np.random.randint(n_clusters + 1, 3 * n_clusters)

                new_Y_train = KMeans(n_clusters=k).fit_predict(X_train)

            sidx = np.argsort(new_Y_train)
            X_train = X_train[sidx,:]
            Y_train = Y_train[sidx]
            new_Y_train = new_Y_train[sidx]

            sample_label_pairs.append([X_train, new_Y_train])
            partition_quality_labels.append(partition_quality)

    testing_examples = dict()

    ridx = np.array(list(range(X_test.shape[0])))
    random.shuffle(ridx)
    X_test = X_test[ridx, :]
    Y_test = Y_test[ridx]
    sidx = np.argsort(Y_test)
    X_test = X_test[sidx,:]
    Y_test = Y_test[sidx]
    testing_examples['correct'] = [np.array(X_test), np.array(Y_test)]

    new_Y_test = KMeans(n_clusters=2).fit_predict(X_test)
    sidx = np.argsort(new_Y_test)
    X_test = X_test[sidx,:]
    Y_test = Y_test[sidx]
    new_Y_test = new_Y_test[sidx]
    testing_examples['under'] = [np.array(X_test), np.array(new_Y_test)]

    new_Y_test = KMeans(n_clusters=n_clusters+2).fit_predict(X_test)
    sidx = np.argsort(new_Y_test)
    X_test = X_test[sidx,:]
    Y_test = Y_test[sidx]
    new_Y_test = new_Y_test[sidx]
    testing_examples['over'] = [np.array(X_test), np.array(new_Y_test)]

    return sample_label_pairs, partition_quality_labels, testing_examples


if __name__ == '__main__':

    sample_label_pairs, partition_quality_labels, testing_examples = generate_data()

    micvi = Meta_iCVI(window_size=15)
    micvi.fit(sample_label_pairs,partition_quality_labels)

    for partition_quality, (X_test, Y_test) in testing_examples.items():
        predicition_history = []
        for sample, label in zip(X_test[:,None],Y_test):
            quality_pred = micvi.increment(sample,label, numeric_prediction=True)
            predicition_history.append(quality_pred)
        plt.figure()
        plt.hold = True
        plt.plot(micvi.correlation_history,'r-')
        plt.plot(predicition_history,'g-')
        plt.title('{} partition example'.format(partition_quality))
        plt.xlabel('Sample Number')
        plt.ylabel('Correlation / Prediction')
    plt.show()






