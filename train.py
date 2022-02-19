import pandas as pd
from ncf import get_model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp


data = pd.read_csv('./data/ratings.csv')
data['interaction'] = data['rating'] > 3
data.replace({False: 0, True: 1}, inplace=True)

movie_id_to_new_id = dict()
id = 1
for index, row in data.iterrows():
    if movie_id_to_new_id.get(row['movieId']) is None:
        movie_id_to_new_id[row['movieId']] = id
        data.at[index, 'movieId'] = id
        id += 1
    else:
        data.at[index, 'movieId'] = movie_id_to_new_id.get(row['movieId'])

train, test = train_test_split(data, test_size=0.2, random_state=42)

train_dataset_12 = tf.data.Dataset.from_tensor_slices((train.userId, train.movieId))
train_dataset_label = tf.data.Dataset.from_tensor_slices(train.interaction)
test_dataset_12 = tf.data.Dataset.from_tensor_slices((test.userId, test.movieId))
test_dataset_label = tf.data.Dataset.from_tensor_slices(test.interaction)

train_dataset = tf.data.Dataset.zip((train_dataset_12, train_dataset_label)).batch(32)
test_dataset = tf.data.Dataset.zip((test_dataset_12, test_dataset_label)).batch(32)

n_users = len(set(data.userId))
n_items = len(set(data.movieId))

HP_HIDDEN = hp.HParam('hidden', hp.Discrete([50, 75, 100]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_LR = hp.HParam('lr', hp.Discrete([.1, .01, .001]))



def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        train_test_model(hparams, run_dir)


def train_test_model(hparams, logdir):
    ncf_model = get_model(n_items, n_users, hparams[HP_DROPOUT], hparams[HP_HIDDEN])

    ncf_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=hparams[HP_LR]),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    ncf_model._name = "neural_collaborative_filtering"
    ncf_model.fit(train_dataset, validation_data = test_dataset, epochs=5,
        callbacks=[tf.keras.callbacks.TensorBoard(logdir), hp.KerasCallback(logdir, hparams)]

     )

session_num = 0

for hidden_layers in HP_HIDDEN.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for lr in HP_LR.domain.values:
      hparams = {
          HP_HIDDEN: hidden_layers,
          HP_DROPOUT: dropout_rate,
          HP_LR: lr,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      #print({h.name: hparams[h] for h in hparams})
      run('logs/hparam_tuning/' + run_name, hparams)
      session_num += 1
