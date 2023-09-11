import copy
import numpy
import pandas as pd

class dataPreProcessingEventEmitter():
    def __init__(self):
        self.events = {}
        self.seed = None
        self.copy = None
        self.train = None
        self.val = None
        self.test = None


    def add_pre_processing(self, event_name: str, obj):
        if not event_name in self.events:
            self.events[event_name] = []
        obj.parent = self
        assert obj.transform in ('all', 'train', 'val', 'test'), "transformation's .transform attribute must be one of: 'all', 'train', 'val, 'test"
        self.events[event_name].append(obj)

    def set_seed_for_all(self, seed):
        self.seed = seed

    def apply(self, event_name, X, y, categorical_indicator, attribute_names):
        if not event_name in self.events:
            return X, y, categorical_indicator, attribute_names

        assert isinstance(X, (pd.Series, pd.DataFrame)), "X must be a Pandas Series or DataFrame"
        assert isinstance(y, (pd.Series, pd.DataFrame)), "Y must be a Pandas Series or DataFrame"

        self.copy = (copy.deepcopy(X), copy.deepcopy(y), copy.deepcopy(categorical_indicator), copy.deepcopy(attribute_names))


        if self.train is None:
            self.train = (X, y, categorical_indicator, attribute_names)

        try:
            for obj in self.events[event_name]:
                if getattr(obj, 'special', False):
                    obj.apply()
                    continue

                datasets = {'train': self.train, 'val': self.val, 'test': self.test}
                for dataset_name, dataset in datasets.items():
                    if obj.transform == 'all' or obj.transform == dataset_name:
                        if dataset is None:
                            continue

                        X, y, categorical_indicator, attribute_names = dataset
                        X, y, categorical_indicator, attribute_names = obj.apply(X, y, categorical_indicator,
                                                                                 attribute_names)
                        datasets[dataset_name] = (X, y, categorical_indicator, attribute_names)

                    elif obj.transform == dataset_name:
                        if dataset is None:
                            raise ValueError(
                                f"The dataset for {dataset_name} is None, but the obj.transform is set to {obj.transform}, have you forgotten to add splitTrainTest or splitTrainValTest?")

                        X, y, categorical_indicator, attribute_names = dataset
                        X, y, categorical_indicator, attribute_names = obj.apply(X, y, categorical_indicator,
                                                                                 attribute_names)
                        datasets[dataset_name] = (X, y, categorical_indicator, attribute_names)

                self.train, self.val, self.test = datasets['train'], datasets['val'], datasets['test']

            self.copy = None


        except Exception as e:
            X = copy.deepcopy(self.copy[0])
            y = copy.deepcopy(self.copy[1])
            categorical_indicator = copy.deepcopy(self.copy[2])
            attribute_names = copy.deepcopy(self.copy[3])
            print(f"Exception occurred: {e}")
            raise




