from src.utils import load_dict_array_list

from copy import deepcopy


class ExperimentResultsArray:
    def __init__(self, dataset, coefs):
        if len(dataset) != len(coefs):
            raise ValueError("coefs and dataset have different len")
        self.dataset = dataset
        self.coefs = coefs

    def __getitem__(self, index):
        result = deepcopy(self.dataset[index])
        result["coef"] = deepcopy(self.coefs[index])
        return result

    def __len__(self):
        return len(self.dataset)


class ModelsExperimentResults:
    def __init__(self, dataset, path_to_coefs):
        self.dataset = dataset
        self.coefs_dict = load_dict_array_list(path_to_coefs)
        self.model_names = self.coefs_dict.keys()

    def get_results(self, model_name) -> ExperimentResultsArray:
        if model_name not in self.model_names:
            raise ValueError("Unknown model name")

        arr = ExperimentResultsArray(self.dataset, self.coefs_dict[model_name])

        return arr
