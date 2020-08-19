# coding=utf-8

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np

from diffnet.evaluation.model_evaluation import evaluate


from diffnet.data.load_data import num_users, num_items



def score_func(batch_user_indices, batch_item_indices):
    return np.random.randn(len(batch_user_indices), len(batch_item_indices))


evaluate(score_func)
