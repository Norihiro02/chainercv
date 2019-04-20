import argparse
import matplotlib.pyplot as plt
%matplotlib inline
import chainer

from chainercv.datasets import camvid_label_colors
from chainercv.datasets import camvid_label_names
from chainercv.links import SegNetBasic
from chainercv import utils
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_semantic_segmentation

#メイン関数
def main():
    chainer.config.train = False
    #chainerの設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)#GPUの使用の可否
    parser.add_argument('--pretrained-model', default='camvid')#訓練済みモデル
    parser.add_argument('image')#読み込み画像
    args = parser.parse_args()
    #Segnetのモデル生成
    model = SegNetBasic(
        n_class=len(camvid_label_names),#ラベルを取得
        pretrained_model=args.pretrained_model)#訓練済みモデルの読み込み
    #GPUの使用
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)#画像の読み込み
    labels = model.predict([img])#セグメンテーチョンを実行
    label = labels[0]#予測結果(整数ラベルのリスト)

    #描画
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    vis_image(img, ax=ax1)
    ax2 = fig.add_subplot(1, 2, 2)
    # Do not overlay the label image on the color image
    vis_semantic_segmentation(
        None, label, camvid_label_names, camvid_label_colors, ax=ax2)
    plt.show()


if __name__ == '__main__':
    main()
