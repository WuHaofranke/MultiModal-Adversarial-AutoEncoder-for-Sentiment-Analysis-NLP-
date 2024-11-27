import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.layers import Input, Bidirectional, Dense, Conv1D, LSTM, MultiHeadAttention, GlobalAveragePooling1D, Concatenate, UpSampling1D, Reshape, Add, LayerNormalization, Dropout
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import regularizers


def visualize_learned_features(model, data, labels, title='Learned Features Visualization'):
    # 提取数据的特征表示
    features = model.encoder.predict(data)

    # 使用 t-SNE 将高维特征映射到二维空间
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
class MMAAE():
    def __init__(self):
        self.latent_dim = 256
        self.v_shape = (10, 512)
        self.a_shape = (128, 512)
        self.t_shape = (36, 768)

        [self.encoder, self.decoder,
         self.discriminator, self.classifier,
         self.enc_disc, self.enc_dec, self.enc_cla] = self.build_mmaae()

        # 使用更小的学习率
        self.encoder.compile(loss='mse', optimizer=AdamW(learning_rate=0.0001, weight_decay=0.005), metrics=['mse'])
        self.enc_dec.compile(loss='mse', optimizer=AdamW(learning_rate=0.0001, weight_decay=0.005), metrics=['mse'])
        self.discriminator.compile(loss='binary_crossentropy', optimizer=AdamW(learning_rate=0.0001, weight_decay=0.005),
                                   metrics=['acc'])
        self.enc_disc.compile(loss='binary_crossentropy', optimizer=AdamW(learning_rate=0.0001, weight_decay=0.005),
                              metrics=['acc'])
        # 使用更小的学习率
        self.enc_cla.compile(loss='sparse_categorical_crossentropy', optimizer=AdamW(learning_rate=0.0001, weight_decay=0.005),
                              metrics=['acc'])

    def build_encoder(self):
        vis_ipt = Input(self.v_shape, name='vis')
        vis_h = Conv1D(128, 3, 1, 'same')(vis_ipt)
        vis_h = Conv1D(128, 1, 1, 'same')(vis_h)
        vis_h = Conv1D(128, 3, 1, 'same')(vis_h)
        vis_h = LSTM(128, activation='relu')(vis_h)

        aud_ipt = Input(self.a_shape, name='aud')
        aud_h = Conv1D(128, 3, 2)(aud_ipt)
        aud_h = LSTM(128)(aud_h)

        tex_ipt = Input(self.t_shape, name='tex')
        tex_q = Conv1D(128, 3, 1)(tex_ipt)
        tex_v = Conv1D(128, 3, 1)(tex_ipt)
        tex_qv_attention = tf.keras.layers.Attention()([tex_q, tex_v])
        tex_q = GlobalAveragePooling1D()(tex_q)
        tex_qv_attention = GlobalAveragePooling1D()(tex_qv_attention)

        h = Concatenate()([vis_h, aud_h, tex_q, tex_qv_attention])
        h = Dense(self.latent_dim)(h)

        encoder = tf.keras.Model(inputs=[vis_ipt, aud_ipt, tex_ipt], outputs=h, name='encoder')
        return encoder

    def build_decoder(self):
        h = Input((self.latent_dim,), name='h_dec')
        rec = Dense(512, activation='relu')(h)
        rec = Dense(2560, activation='relu')(h)
        rec = Reshape((5, 512))(rec)
        rec = tf.keras.layers.UpSampling1D(size=2)(rec)
        audio_feature = Conv1D(1, kernel_size=1, activation='linear')(rec)
        decoder = tf.keras.Model(h, rec, name='decoder')
        return decoder

    def build_classifier(self):
        h = Input((self.latent_dim,), name='h_cla')
        res = Dense(3, activation='softmax')(h)
        classifier = tf.keras.Model(h, res, name='classifier')
        return classifier

    def build_discriminator(self):
        h = Input((self.latent_dim,), name='h_disc')
        res = Dense(1, activation='sigmoid')(h)
        discriminator = tf.keras.Model(h, res, name='disc')
        return discriminator

    def build_mmaae(self):
        encoder = self.build_encoder()
        decoder = self.build_decoder()
        discriminator = self.build_discriminator()
        classifier = self.build_classifier()

        vis_ipt = Input(self.v_shape, name='vis_mmaae')
        aud_ipt = Input(self.a_shape, name='aud_mmaae')
        tex_ipt = Input(self.t_shape, name='tex_mmaae')

        h = encoder([vis_ipt, aud_ipt, tex_ipt])
        validity = discriminator(h)
        emotion = classifier(h)
        reconstruction = decoder(h)

        enc_disc = tf.keras.Model(inputs=[vis_ipt, aud_ipt, tex_ipt], outputs=validity)
        enc_dec = tf.keras.Model(inputs=[vis_ipt, aud_ipt, tex_ipt], outputs=reconstruction)
        enc_cla = tf.keras.Model(inputs=[vis_ipt, aud_ipt, tex_ipt], outputs=emotion)

        return [encoder, decoder, discriminator, classifier, enc_disc, enc_dec, enc_cla]
