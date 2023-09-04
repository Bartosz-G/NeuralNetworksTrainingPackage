import copy
import numpy
import pandas as pd

class dataPreProcessingEventEmitter():
    def __init__(self):
        self.events = {}
        self.seed = None
        self.X_copy = None
        self.y_copy = None
        self.categorical_indicators_copy = None
        self.attribute_names_copy = None


    def add_pre_processing(self, event_name: str, obj):
        if not event_name in self.events:
            self.events[event_name] = []
        obj.parent = self
        self.events[event_name].append(obj)

    def set_seed_for_all(self, seed):
        self.seed = seed

    def apply(self, event_name, X, y, categorical_indicator, attribute_names):
        if not event_name in self.events:
            return X, y, categorical_indicator, attribute_names

        assert isinstance(X, (pd.Series, pd.DataFrame)), "X must be a Pandas Series or DataFrame"
        assert isinstance(y, (pd.Series, pd.DataFrame)), "Y must be a Pandas Series or DataFrame"

        self.X_copy = copy.deepcopy(X)
        self.y_copy = copy.deepcopy(y)
        self.categorical_indicators_copy = copy.deepcopy(categorical_indicator)
        self.attribute_names_copy = copy.deepcopy(attribute_names)

        try:
            for obj in self.events[event_name]:
                X, y, categorical_indicator, attribute_names = obj.apply(X, y, categorical_indicator, attribute_names)

            return X, y, categorical_indicator, attribute_names

        except Exception as e:
            X = self.X_copy
            y = self.y_copy
            categorical_indicator = self.categorical_indicators_copy
            attribute_names = self.attribute_names_copy
            raise




