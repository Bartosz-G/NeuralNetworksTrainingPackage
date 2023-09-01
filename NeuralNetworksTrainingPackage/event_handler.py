import copy

class dataPreProcessingEventEmitter():
    def __init__(self):
        self.events = {}
        self.X_copy = None
        self.y_copy = None
        self.categorical_indicators_copy = None
        self.attribute_names_copy = None


    def addPreProcessing(self, event_name: str, obj):
        if not event_name in self.events:
            self.events[event_name] = []
        self.events[event_name].append(obj)

    def apply(self, event_name, X, y, categorical_indicator, attribute_names):
        if not event_name in self.events:
            return False

        self.X_copy = copy.deepcopy(X)
        self.y_copy = copy.deepcopy(y)
        self.categorical_indicators_copy = copy.deepcopy(categorical_indicator)
        self.attribute_names_copy = copy.deepcopy(attribute_names)

        try:
            for obj in self.events[event_name]:
                X, y, categorical_indicator, attribute_names = obj.transform(X, y, categorical_indicator, attribute_names)

        except Exception as e:
            X = self.X_copy
            y = self.y_copy
            categorical_indicator = self.categorical_indicators_copy
            attribute_names = self.attribute_names_copy
            raise




