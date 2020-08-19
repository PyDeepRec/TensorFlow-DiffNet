# coding=utf-8

import tensorflow as tf

from diffnet.config import embedding_size

tf.enable_eager_execution()

from diffnet.data.load_data import *
from diffnet.evaluation.model_evaluation import evaluate


batch_size = 2000

user_embeddings = tf.Variable(tf.random.truncated_normal([num_users, embedding_size], stddev=np.sqrt(1/embedding_size)))
item_embeddings = tf.Variable(tf.random.truncated_normal([num_items, embedding_size], stddev=np.sqrt(1/embedding_size)))


fc = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
])



# optimizer = tf.keras.optimizers.Adam(learning_rate=3e-3)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=10.0)
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)


def forward(embedded_users, embedded_items, perm):
    if perm:
        return embedded_users @ tf.transpose(embedded_items, [1, 0])
    else:
        return tf.reduce_sum(embedded_users * embedded_items, axis=-1)


def forward2(embedded_users, embedded_items, perm):
    if perm:
        num_batch_users = embedded_users.shape[0]
        num_batch_items = embedded_items.shape[0]
        embedded_users = tf.expand_dims(embedded_users, axis=1)
        embedded_items = tf.expand_dims(embedded_items, axis=0)

        embedded_users = tf.tile(embedded_users, [1, num_batch_items, 1])
        embedded_items = tf.tile(embedded_items, [num_batch_users, 1, 1])

    interaction = embedded_users * embedded_items

    # logits = tf.reduce_sum(interaction, axis=-1) + tf.squeeze(fc(interaction), axis=-1)

    logits = tf.squeeze(fc(interaction), axis=-1)

    return logits


def mf_score_func(batch_user_indices, batch_item_indices):
    embedded_users = tf.nn.embedding_lookup(user_embeddings, batch_user_indices)
    embedded_items = tf.nn.embedding_lookup(item_embeddings, batch_item_indices)
    logits = forward(embedded_users, embedded_items, perm=True)
    return logits



for epoch in range(1000):

    for step, batch_edges in enumerate(tf.data.Dataset.from_tensor_slices(train_user_item_edges).shuffle(1000000).batch(batch_size)):
        batch_user_indices = batch_edges[:, 0]
        batch_item_indices = batch_edges[:, 1]

        batch_neg_item_indices = np.random.randint(0, num_items, batch_item_indices.shape)

        # batch_neg_item_indices = []
        # for user_index in batch_user_indices.numpy():
        #     train_items = train_user_items_dict[user_index]
        #     while True:
        #         neg_item_index = np.random.randint(0, num_items)
        #         if neg_item_index in train_items:
        #             continue
        #         else:
        #             batch_neg_item_indices.append(neg_item_index)
        #             break
        # batch_neg_item_indices = np.array(batch_neg_item_indices)


        with tf.GradientTape() as tape:
            embedded_users = tf.nn.embedding_lookup(user_embeddings, batch_user_indices)
            embedded_items = tf.nn.embedding_lookup(item_embeddings, batch_item_indices)
            embedded_neg_items = tf.nn.embedding_lookup(item_embeddings, batch_neg_item_indices)

            pos_logits = forward(embedded_users, embedded_items, perm=False)
            neg_logits = forward(embedded_users, embedded_neg_items, perm=False)


            pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pos_logits,
                labels=tf.ones_like(pos_logits)
            )

            neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=neg_logits,
                labels=tf.zeros_like(pos_logits)
            )

            losses = pos_losses + neg_losses

            l2_vars = [user_embeddings, item_embeddings]
            l2_losses = [tf.nn.l2_loss(var) for var in l2_vars]
            l2_loss = tf.add_n(l2_losses)
            loss = tf.reduce_sum(losses) + l2_loss * 1e-2

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))


        if step % 1000 == 0:

            print(epoch, step, loss)

        if epoch > 0 and epoch % 20 == 0 and step == 0:
            evaluate(mf_score_func)



