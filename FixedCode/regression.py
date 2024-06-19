from base import Base
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model, model_from_json
from keras.layers import (Dense, Dropout, Embedding,
                          Activation, Input, concatenate, Reshape, Flatten)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from metrics import r2
import keras


class Embedder(Base):

    def __init__(self, emb_sizes, model_json=None):

        super(Embedder, self).__init__(emb_sizes, model_json)

    def fit(self, X, y, mode="train",
            batch_size=256, epochs=100,
            checkpoint=None,
            early_stop=None):
        '''
        Fit a neural network on the data.

        :param X: input DataFrame
        :param y: input Series
        :param batch_size: size of mini-batch
        :param epochs: number of epochs for training
        :param checkpoint: optional Checkpoint object
        :param early_stop: optional EarlyStopping object
        :return: Embedder instance
        '''
        # change learning rate
        optimizer = Adam(lr=0.0001)

        nnet = self._create_model(X, model_json=self.model_json)

        nnet.compile(loss='mean_squared_error',
                     optimizer=optimizer,
                     metrics=[r2])

        callbacks = list(filter(None, [checkpoint, early_stop]))
        callbacks = callbacks if callbacks else None

        x_inputs_list = self._prepare_inputs(X)

        if mode == "train":
          nnet.fit(x_inputs_list, y.values, batch_size=batch_size,
                  epochs=epochs,
                  callbacks=callbacks,
                  validation_split=0.3)
        elif mode == "test":
          nnet.fit(x_inputs_list, y.values, batch_size=batch_size,
                  epochs=epochs,
                  callbacks=callbacks,
                  validation_split=0)

        self.model = nnet

        return self

    def fit_transform(self, X, y,
                      batch_size=256, epochs=100,
                      checkpoint=None,
                      early_stop=None,
                      as_df=False
                      ):
        '''
        Fit a neural network and transform the data.

        :param X: input DataFrame
        :param y: input Series
        :param batch_size: size of mini-batch
        :param epochs: number of epochs for training
        :param checkpoint: optional Checkpoint object
        :param early_stop: optional EarlyStopping object
        :return: transformed data
        '''
        self.fit(X, y, batch_size, epochs,
                 checkpoint, early_stop)

        return self.transform(X, as_df=as_df)

    def _default_nnet(self, X):

        emb_sz = self.emb_sizes
        numerical_vars = [x for x in X.columns
                          if x not in self._categorical_vars]

        inputs = []
        flatten_layers = []

        for var, sz in emb_sz.items():
            input_c = Input(shape=(1,), dtype='int32')
            embed_c = Embedding(*sz, input_length=1)(input_c)
            # embed_c = Dropout(0.25)(embed_c)
            flatten_c = Flatten()(embed_c)

            inputs.append(input_c)
            flatten_layers.append(flatten_c)

        input_num = Input(shape=(len(numerical_vars),), dtype='float32')
        flatten_layers.append(input_num)
        inputs.append(input_num)

        flatten = concatenate(flatten_layers, axis=-1)

        fc1 = Dense(1000, kernel_initializer='normal')(flatten)
        fc1 = Activation('relu')(fc1)
        # fc1 = BatchNormalization(fc1)
        # fc1 = Dropout(0.75)(fc1)

        fc2 = Dense(500, kernel_initializer='normal')(fc1)
        fc2 = Activation('relu')(fc2)
        # fc2 = BatchNormalization(fc2)
        # fc2 = Dropout(0.5)(fc2)

        output = Dense(1, kernel_initializer='normal')(fc2)
        nnet = Model(inputs=inputs, outputs=output)

        return nnet