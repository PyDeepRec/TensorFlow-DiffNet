# coding=utf-8

# from ngcf.data.load_adj_data import *

import numpy as np
import os
from tqdm import tqdm

from diffnet.config import data_dir, train_path, test_path, cache_path
from diffnet.utils.data_utils import load_cache




def read_user_item_info(file_path):

    raw_user_item_edges = []
    user_items_dict = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):

            items = [int(item) for item in line.split()[:3]]
            user_index = items[0]
            item_index = items[1]
            rating = items[2]

            if user_index not in user_items_dict:
                item_set = set()
                user_items_dict[user_index] = item_set
            else:
                item_set = user_items_dict[user_index]

            if rating > 0:
                item_set.add(item_index)
                raw_user_item_edges.append([user_index, item_index])


    raw_user_item_edges = np.array(raw_user_item_edges)
    return user_items_dict, raw_user_item_edges


def read_data():
    train_user_items_dict, train_user_item_edges = read_user_item_info(train_path)
    test_user_items_dict, test_user_item_edges = read_user_item_info(test_path)
    return train_user_items_dict, test_user_items_dict, train_user_item_edges, test_user_item_edges, test_user_neg_items_dict


train_user_items_dict, test_user_items_dict, train_user_item_edges, test_user_item_edges, test_user_neg_items_dict \
    = load_cache(cache_path, create_func=read_data)


raw_user_item_edges = np.concatenate([train_user_item_edges, test_user_item_edges], axis=0)
num_users, num_items = raw_user_item_edges.max(axis=0) + 1



# print(test_user_neg_items_dict)