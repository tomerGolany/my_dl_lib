import numpy as np


class GenericDataSetIterator:

    def __init__(self, data, labels):
        """

        :param data:
        """
        self.data = data
        self.labels = labels
        self.num_of_examples = data.shape[0]
        self.iterator_epoch_num = 0
        self.iterator_index_in_epoch = 0
        self.shuffled_data = data
        self.shuffled_labels = labels

    def next_batch(self, batch_size, shuffle=True):
        """
        returns a batch of size batch_size from the data, if shuffle=True it shuffels it.
        :param batch_size: size of batch to be returned
        :param shuffle: True/False
        :return: returns a batch of size batch_size from the data, if shuffle=True is shuffels it.
        """
        start = self.iterator_index_in_epoch
        if shuffle:
            # Only run once:
            if start == 0 and self.iterator_epoch_num == 0:
                # First iteration and first epoch --> shuffle all elements:
                idx = np.arange(0, self.num_of_examples)  # ==> [0,1,2,...self.num_of_examples - 1]
                np.random.shuffle(idx)  # shuffle indexes
                self.shuffled_data = self.data[idx]
                self.shuffled_labels = self.labels[idx]

        # Get next Batch:
        if start + batch_size > self.num_of_examples:
            # Case 1:  Go to next epoch:
            self.iterator_epoch_num += 1
            # Get of all the examples left in the last epoch:
            data_left_from_last_epoch = self.shuffled_data[start:self.num_of_examples]
            labels_left_from_last_epoch = self.shuffled_labels[start:self.num_of_examples]
            number_of_examples_took_from_last_epoch = len(data_left_from_last_epoch)

            # Shuffle the data again:
            if shuffle:
                idx = np.arange(0, self.num_of_examples)  # ==> [0,1,2,...self.num_of_examples - 1]
                np.random.shuffle(idx)  # shuffle indexes
                self.shuffled_data = self.data[idx]
                self.shuffled_labels = self.labels[idx]
            start = 0
            self.iterator_index_in_epoch = batch_size - number_of_examples_took_from_last_epoch
            end = self.iterator_index_in_epoch
            data_from_new_epoch = self.shuffled_data[start:end]
            labels_from_new_epoch = self.shuffled_labels[start:end]
            return np.concatenate((data_left_from_last_epoch, data_from_new_epoch), axis=0), \
                   np.concatenate((labels_left_from_last_epoch, labels_from_new_epoch), axis=0)

        else:
            # we are in the same epoch:
            self.iterator_index_in_epoch += batch_size
            end = self.iterator_index_in_epoch
            return self.shuffled_data[start:end], self.shuffled_labels[start:end]

    def convert_labels_to_one_hot_vector(self, num_of_classes):
        """

        :param num_of_classes:
        :return:
        """
        # TODO: Check if we need to implement more efficiently.
        one_hot_tags = []
        for tag in self.labels:
            one_hot = [0 for x in range(num_of_classes)]
            one_hot[tag] = 1
            one_hot_tags.append(one_hot)
        self.labels = np.array(one_hot_tags)
        return np.array(one_hot_tags)


if __name__ == "__main__":
    data = np.arange(0, 10)
    labels = np.arange(1, 11)
    print("Testing Data with 4 batches and 10 iterations: ", data)
    dataset = GenericDataSetIterator(data, labels)
    for i in range(10):
        print("Iteration ", i)
        print(dataset.next_batch(4))


