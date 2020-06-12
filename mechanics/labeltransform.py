import numpy as np
from pycox.preprocessing.label_transforms import LabTransCoxTime



class LabTransFlexible(LabTransCoxTime):
    """
    Similar to pycox label transformer, but with flexible sklearn transformer.
    """
    def __init__(self, transformer, log_duration=False, transformerkwargs={}):

        self.log_duration = log_duration
        self.transformerkwargs = transformerkwargs
        self.duration_scaler = transformer(**transformerkwargs)
