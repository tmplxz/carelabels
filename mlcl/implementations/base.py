class BaseImplementation:

    def __init__(self):
        self.config = {}
        self.ds_class = None
        self.cl_file = None

    def name(self):
        return 'Default Implementation'

    # METHOD TO OVERRIDE:

    def prepare(self, args):
        raise NotImplementedError

    def get_info(self):
        raise NotImplementedError

    def get_meta_info(self):
        raise NotImplementedError
    
    def train(self, X, y=None, ds_info=None):
        # for generative models, no y should be required
        raise NotImplementedError

    def apply(self, X, ds_info=None):
        raise NotImplementedError
