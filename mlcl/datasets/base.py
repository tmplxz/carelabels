class DatasetBase:

    def __init__(self, cfg):
        pass

    def get_train(self):
        raise NotImplementedError

    def get_apply(self):
        raise NotImplementedError

    def info(self):
        raise NotImplementedError

    def evaluate(self, prediction):
        raise NotImplementedError
