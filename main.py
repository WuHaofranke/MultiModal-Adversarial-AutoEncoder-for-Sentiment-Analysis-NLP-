from MMAAE import MMAAE,visualize_learned_features
from plot_results import plot_results
from utilz import load_features
import numpy as np


def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def train(epoch, batch_size, data, models):
    [visual_clip, acoustic, bert_embs, label] = data
    val_res, val_acc, tes_res, tes_acc = [], [], [], []

    for e in range(epoch):
        print('----------Epoch: %d----------' % e)
        iters = int(len(label['train']) / batch_size)

        for it in range(iters):
            # 交替训练自编码器、分类器和判别器
            if it % 3 == 0:
                # 训练自编码器
                set_trainable(models.encoder, True)
                set_trainable(models.enc_dec, True)
                set_trainable(models.enc_cla, False)
                set_trainable(models.discriminator, False)
                set_trainable(models.enc_disc, False)
                idx = np.random.randint(0, len(label['train']), batch_size)
                [batch_vis, batch_aud, batch_tex, batch_labels] = [
                    np.array(visual_clip['train'])[idx],
                    np.array(acoustic['train'])[idx],
                    np.array(bert_embs['train'])[idx],
                    np.array(label['train'])[idx]]
                rec_loss = models.enc_dec.train_on_batch([batch_vis, batch_aud, batch_tex], batch_vis)
                print('rec_loss', rec_loss)

            elif it % 3 == 1:
                # 训练分类器
                set_trainable(models.encoder, True)
                set_trainable(models.enc_dec, False)
                set_trainable(models.enc_cla, True)
                set_trainable(models.discriminator, False)
                set_trainable(models.enc_disc, False)
                idx = np.random.randint(0, len(label['train']), batch_size)
                [batch_vis, batch_aud, batch_tex, batch_labels] = [
                    np.array(visual_clip['train'])[idx],
                    np.array(acoustic['train'])[idx],
                    np.array(bert_embs['train'])[idx],
                    np.array(label['train'])[idx]]
                cla_loss = models.enc_cla.train_on_batch([batch_vis, batch_aud, batch_tex], batch_labels)
                print('cla_loss', cla_loss)

            else:
                # 训练判别器
                set_trainable(models.encoder, False)
                set_trainable(models.enc_dec, False)
                set_trainable(models.enc_cla, False)
                set_trainable(models.discriminator, True)
                set_trainable(models.enc_disc, True)
                idx = np.random.randint(0, len(label['train']), batch_size)
                [batch_vis, batch_aud, batch_tex, batch_labels] = [
                    np.array(visual_clip['train'])[idx],
                    np.array(acoustic['train'])[idx],
                    np.array(bert_embs['train'])[idx],
                    np.array(label['train'])[idx]]
                batch_pred_h = models.encoder.predict([batch_vis, batch_aud, batch_tex])
                latent_normal = np.random.normal(size=(batch_size, 256))
                dis_loss = models.discriminator.train_on_batch(
                    np.concatenate([batch_pred_h, latent_normal], 0),
                    np.concatenate([np.ones(batch_size), np.zeros(batch_size)], 0))
                print('dis_loss', dis_loss)

        # 在每个epoch结束后进行评估
        enc_dis_val_loss, enc_dis_val_acc = models.enc_cla.evaluate(
            [np.array(visual_clip['valid']), np.array(acoustic['valid']), np.array(bert_embs['valid'])],
            np.array(label['valid']), verbose=0)
        val_res.append(enc_dis_val_loss)
        val_acc.append(enc_dis_val_acc)
        print('Validation Loss:', enc_dis_val_loss)
        print('Validation Accuracy:', enc_dis_val_acc)

        enc_dis_tes_loss, enc_dis_tes_acc = models.enc_cla.evaluate(
            [np.array(visual_clip['test']), np.array(acoustic['test']), np.array(bert_embs['test'])],
            np.array(label['test']), verbose=0)
        tes_res.append(enc_dis_tes_loss)
        tes_acc.append(enc_dis_tes_acc)
        print('Test Loss:', enc_dis_tes_loss)
        print('Test Accuracy:', enc_dis_tes_acc)

        print('----------Epoch: %d----------')

    # 绘制损失值和精确度曲线
    plot_results(val_res, val_acc, tes_res, tes_acc, epoch)

if __name__ == '__main__':
    # 加载数据集和模型
    visual_clip = load_features("./data/visual_clip.pkl")
    acoustic = load_features("./data/acoustic_wav2vec.pkl")
    bert_embs = load_features("./data/textual_bert.pkl")
    label = load_features("./data/labels.pkl")
    data = [visual_clip, acoustic, bert_embs, label]
    models = MMAAE()

    # 训练模型
    train(epoch=11, batch_size=16, data=data, models=models)
  