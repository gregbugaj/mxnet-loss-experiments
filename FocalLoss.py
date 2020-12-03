from mxnet.gluon.loss import Loss
import numpy as np

class FocalLoss(Loss):
    def __init__(self, axis=-1, weight=None, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        print('Focal loss ********* ')
        pred = label
        print(pred.shape)
        print(label.shape)
        loss = 1.0
        
        print('loss = %s' %(loss))
        return loss
