# coding=utf-8
import os

# replace data_base_dir with path of your own data directory
data_base_dir = os.path.join(os.path.dirname(__file__), "../data")
dataset = "yelp"
data_dir = os.path.join(data_base_dir, dataset)


train_path = os.path.join(data_dir, "{}.train.rating".format(dataset))
test_path = os.path.join(data_dir, "{}.test.rating".format(dataset))
links_path = os.path.join(data_dir, "{}.links".format(dataset))
cache_path = os.path.join(data_dir, "cache.p")
# test_neg_path = os.path.join(data_dir, "{}.test.negative".format(dataset))

print(os.listdir(data_dir))
