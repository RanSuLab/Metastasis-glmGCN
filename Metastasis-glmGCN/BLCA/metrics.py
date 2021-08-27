import tensorflow as tf
import numpy as np
np.random.seed(123)
tf.set_random_seed(123)
def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    y_pred = tf.argmax(preds, 1)

    y_score=preds
    y_actual = tf.argmax(labels, 1)
    return tf.reduce_mean(accuracy_all),y_pred,y_actual,y_score

def confusionmetrics(y_pred,y_actual):
    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(len(y_pred)):
        if int(y_pred[i])==1 and int(y_actual[i]) == 1:
            TP += 1
        elif int(y_pred[i])==1 and int(y_actual[i]) == 0:
            FP += 1
        elif int(y_pred[i])==0 and int(y_actual[i]) == 1:
            FN += 1
        elif int(y_pred[i])==0  and int(y_actual[i]) == 0:
            TN += 1
    accuracy = (TP +TN) / (TP + FP + FN + TN)
    tpr = TP/ (TP + FN)
    fpr = FP / (FP + TN)
    fnr = FN/ (TP + FN)
    tnr = TN / (FP + TN)
    specificity=tpr
    recall=tpr
    sensitivity=tnr
    precision =TP / (TP  + FP)
    f1_score = (2 * (precision * recall)) / (precision + recall)
    return specificity,sensitivity, f1_score, accuracy
