"""
Evaluation metrics.
"""
from pycox.evaluation.concordance import concordance_td
from sksurv.metrics import concordance_index_ipcw, _estimate_concordance_index






def concordance_index_ipcw(survival_train, survival_test, estimate, tau=None, tied_tol=1e-8):
    """
    An implementation of the ipcw corrected Cidx, following sksurv, but w/o structured np arrays.

    instead survival_train and survival_test are ndim arrays.

    :param survival_train:
    :param survival_test:
    :param estimate:
    :param tau:
    :param tied_tol:
    :return:
    """

    return _estimate_concordance_index(test_event, test_time, estimate, w, tied_tol)



