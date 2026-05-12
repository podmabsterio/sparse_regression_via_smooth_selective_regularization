class BaseDatasets:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        X_train, y_train, theta_true, X_val, y_val = self.data[index]
        dataset_dict = {
            "X": X_train,
            "y": y_train,
            "X_test": X_val,
            "y_test": y_val,
            "theta_true": theta_true,
            "index": index,
        }
        return dataset_dict

    def __len__(self):
        return len(self.data)
