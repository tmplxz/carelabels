class BaseImplementation:

    def __init__(self):
        self.config = {}
        self.ds_class = None
        self.cl_file = None

    # METHOD TO OVERRIDE:

    def prepare(self, args):
        raise NotImplementedError

    def get_info(self):
        # info dict has to contain fields 'config_id' (str value) and 'uses_gpu' (bool value)
        raise NotImplementedError

    def get_meta_info(self):
        raise NotImplementedError
    
    def train(self, data, ds_info=None):
        raise NotImplementedError 

    def apply(self, data, ds_info=None):
        raise NotImplementedError
