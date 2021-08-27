from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import auc, roc_curve
from tensorflow.python.keras.utils import to_categorical
import pandas as pd

import utils
from models_glmGCN import glmGCN
from utils import  sample_mask, construct_feed_dict,preprocess_Finaladj,preprocess_features
# STAD_glmGCN
# train_ [1.0, 0.99999994, 1.0, 1.0, 0.99999994, 0.99999994, 1.0, 1.0, 1.0, 1.0]
# test_ [0.9857143, 0.9714285, 0.9714285, 0.98571426, 0.98571426, 0.9714285, 0.98571426, 0.9705882, 0.9852942, 0.9264706]
# train_ 1.0 test_ 0.9739496
# specificity 0.9826050420168068 sensitivity 0.9652941176470587  f1_score 0.9736027871088062 auc 0.9926568203038791
# STAD_glmGCN
# train_ [1.0, 0.99999994, 1.0, 1.0, 0.99999994, 0.99999994, 1.0, 1.0, 1.0, 1.0]
# test_ [0.9857143, 0.98571426, 0.98571426, 1.0, 0.9714285, 0.9714285, 0.98571426, 0.9558823, 0.9852942, 0.9117646]
# train_ 1.0 test_ 0.9738655
# specificity 0.9826050420168068 sensitivity 0.965126050420168  f1_score 0.9733738684259199 auc 0.9925600543247602
# STAD_glmGCN
# train_ [1.0, 0.99999994, 1.0, 1.0, 0.99999994, 0.99839735, 1.0, 1.0, 1.0, 1.0]
# test_ [0.9857143, 0.9714285, 0.98571426, 1.0, 0.9714285, 0.9714285, 0.98571426, 0.9558823, 0.9852942, 0.9264706]
# train_ 0.9998398 test_ 0.9739076
# specificity 0.9826050420168068 sensitivity 0.9652100840336134  f1_score 0.9735101898413954 auc 0.9929496647143705


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'glmGCN', 'Model string.')
flags.DEFINE_float('lr1', 0.1, 'Initial  GLM learning rate.')
flags.DEFINE_float('lr2', 0.00005, 'Initial GCN learning rate.')
flags.DEFINE_integer('epochs',1500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden_gcn',25, 'Number of units in GCN hidden layer.')
flags.DEFINE_integer('hidden_gl', 90, 'Number of units in  GLM hidden layer.')
flags.DEFINE_float('dropout',0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-8, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_float('losslr1', 0.01, 'xxxx')
flags.DEFINE_float('losslr2',0.00001, 'xxxx')
flags.DEFINE_float('losslr3',0.00001,'xxxx')
flags.DEFINE_integer('seed',123, 'Number of epochs to train.')
np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
# Load data
# Define model evaluation function
def evaluate(features, secondinput, adj, labels, mask, epoch, placeholders,flag=0):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, secondinput, adj, labels, mask, epoch, placeholders)
    if flag == 0:
        outs_val = sess.run([model.loss, model.accuracy,model.y_pred,model.y_actual,model.y_score,model.layer], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1],  (time.time() - t_test),outs_val[2],outs_val[3],outs_val[4],outs_val[5]
    else:
        outs_val = sess.run(model.accuracy, feed_dict=feed_dict_val)
        return outs_val

data=pd.read_csv('input_STAD.csv')
data.drop('Tags',axis=1,inplace=True)
# x: gene feature(patients as rows and genes as columns )(n*p)
# y: label
y = data['Metastasis']
x=data.drop('Metastasis', axis=1)
# Solve the problem of sample imbalance,
# Analyze a few classes and synthesize new samples to add to the data set according to a few classes
from imblearn.over_sampling import SMOTE
smo = SMOTE(random_state=0)
x, y = smo.fit_sample(x, y)
LEN=x.shape[0]
# Obtain data with rows of genes and columns of patients and normalize p*1
features =preprocess_features(np.transpose(x))
# Convert the category vector to a binary representation
labels = to_categorical(y)
# Extend dimension to n*p*1
second=np.array(x)
secondinput=np.expand_dims(second, -1)
# A=A+I
ppi_matrix = np.load("ppi_STAD.npy")
adj,edge= preprocess_Finaladj(ppi_matrix)

from sklearn.model_selection import StratifiedKFold
KF=StratifiedKFold(n_splits=10,random_state=3,shuffle=True)
count1 = 0
mean_fpr = np.linspace(0, 1, 100)
i = 0
aucs=[]
specificitys=[]
sensitivitys=[]
f1_scores=[]
train_ACC=[]
test_ACC=[]
tprs=[]
tnrs=[]
for train,test in KF.split(x,y):
    count1 = count1 + 1
    loss = {"train": [], "val": []}
    accuracy = {"train_acc": [], "val_acc": []}
    train_mask = sample_mask(train, labels.shape[0])
    test_mask = sample_mask(test, labels.shape[0])
    y_train = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    # Define placeholders
    placeholders = {
        'adj': tf.sparse_placeholder(tf.float32),
        'features': tf.sparse_placeholder(tf.float32,
                                          shape=tf.constant([features[2][0], features[2][1]], dtype=tf.int64)),
        'secondinput': tf.placeholder(shape=[features[2][1], features[2][0], 1], dtype=tf.float32),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_nodes': tf.placeholder(tf.int32, ),
        'step': tf.placeholder(tf.int32),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }
    # num:p input_dim:n
    model = glmGCN(placeholders, edge, adj, num=features[2][0], input_dim=features[2][1], logging=True)
    # # # Initialize session
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'  # Specifies which GPU is available
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # The program allocates memory on demand
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    # Train model
    for epoch in range(FLAGS.epochs):
        t = time.time()
        feed_dict = construct_feed_dict(features, secondinput
                                        , adj, y_train, train_mask, epoch, placeholders, )
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.y_pred, model.y_actual],
                        feed_dict=feed_dict)

        cost, acc, duration, y_pred, y_actual, y_score, layer = evaluate(features, secondinput
                                                                         , adj, y_test, test_mask, epoch, placeholders)
        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
        loss["train"].append(outs[1])
        loss["val"].append(cost)
        accuracy["train_acc"].append(outs[2])
        accuracy["val_acc"].append(acc)

    save_path = saver.save(sess, "STAD_MODEL/model" + str(count1))
    tf.reset_default_graph()
    train_ACC.append(outs[2])
    test_ACC.append(acc)
    yy_pred = []
    yy_score = []
    yy_actual = []
    for i in range(y_pred.shape[0]):
        if test_mask[i] == 1:
            yy_actual.append(y_actual[i])
            yy_pred.append(y_pred[i])
            yy_score.append(y_score[i][1])
    specificity, sensitivity, f1_score, myacc = utils.confusionmetrics(yy_pred, yy_actual)
    specificitys.append(specificity)
    sensitivitys.append(sensitivity)
    f1_scores.append(f1_score)
    print("STAD_glmGCN")
    print("train_", train_ACC)
    print("test_", test_ACC)
    FPRs, TPRs, thresholds = roc_curve(yy_actual, yy_score)
    tprs.append(np.interp(mean_fpr, FPRs, TPRs))
    tprs[-1][0] = 0.0

np.save("STAD_glmGCN.npy", tprs)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
print("train_", np.mean(train_ACC), "test_", np.mean(test_ACC))
print("specificity", np.mean(specificitys), "sensitivity", np.mean(sensitivitys), " f1_score", np.mean(f1_scores),
      "auc", mean_auc)
