import os
import shutil
import time

import gc
import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

env_path = "/usr/local/Caskroom/miniforge/base/envs/joye_env"
font_path = "lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf"
font = ImageFont.truetype(os.path.join(env_path, font_path), 32)


config = {
    "font.family": "serif",
    "font.serif": ["SimHei"]
}

plt.rcParams.update(config)


class InferImage:
    def __init__(self, model=None) -> None:
        # Use GPU if it is available otherwise use CPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        availability = "available" if torch.cuda.is_available() else "unavailable"
        print(f'GPU is {availability} thus using {self.device.type.upper()}')
        if model is None:
            # Load pre-trained image classification models
            model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # set model to evaluation/inference status
        self.model = model.eval()
        # send move to device
        self.model = self.model.to(self.device)

        # 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
        self.test_transform = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
        self.img_path = None
        self.verbose = True
        self.language = 'en'

    def preprocess(self, img_pil, show_plot=True, verbose=True):

        preprocessed_img = self.test_transform(img_pil)
        if not self.verbose:
            print("Before preprocess: ", np.array(img_pil).shape)
            print("After preprocess: ", preprocessed_img.shape)
        if show_plot:
            fig = plt.figure(figsize=(18, 6))
            # Plot the annotated figure on the right hand-side
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(img_pil)
            ax1.axis('off')
            ax2 = plt.subplot(1, 2, 2)
            # ax2.imshow(torch.reshape(preprocessed_img,(224,224,3)))
            # ax2.imshow(preprocessed_img.reshape(224,224,3))
            # ax2.imshow(preprocessed_img.permute(2,1,0))
            # ax2.imshow((preprocessed_img.permute(2,1,0).numpy()*255).astype('uint8'))
            p = preprocessed_img.permute(2, 1, 0).numpy()
            p = p/np.amax(p)
            p = np.clip(p, 0, 1)
            ax2.imshow(p)
            ax2.axis('off')
        return preprocessed_img

    def inference(self, input_img):
        input_img = input_img.unsqueeze(0).to(self.device)
        pred_logits = self.model(input_img)  # Model inference
        # Apply softmax
        pred_softmax = F.softmax(pred_logits, dim=1)

        if not self.verbose:
            print("Run model inference and output logits. The shape is ->",
                  pred_logits.shape)
            print("Apply softmax on the logits to get probability. The shape is ->",
                  pred_softmax.shape)
        return pred_softmax

    def plot_class_probability(self, pred_softmax, img_path=None, bar_label=False):
        if img_path is None:
            if self.img_path is None:
                img_path = ""
            else:
                img_path = self.img_path
        y = pred_softmax.cpu().detach().numpy()[0]
        plt.figure(figsize=(8, 4))

        x = range(y.shape[0])
        ax = plt.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
        plt.ylim([0, 1.0])  # y axis

        if bar_label:
            plt.bar_label(ax, fmt='%.2f', fontsize=15)  # 置信度数值

        if self.language == 'en':
            plt.xlabel('Class', fontsize=10, fontname='Arial')
            plt.ylabel('Confidence', fontsize=10, fontname='Arial')

        else:

            plt.xlabel('类别', fontsize=10)
            plt.ylabel('置信度', fontsize=10)
        plt.tick_params(labelsize=10)
        plt.title(img_path, fontsize=15, fontname='Arial')
        plt.show()

    def get_top_n(self, pred_softmax, n=10):
        top_n = torch.topk(pred_softmax, n)
        # get the id of top n
        pred_ids = top_n[1].cpu().detach().numpy().squeeze()
        # get the probablity of top n
        confs = top_n[0].cpu().detach().numpy().squeeze()
        return pred_ids, confs

    def get_idx_labels(self, file_path=None):
        if file_path is None:
            file_path = 'data/image_net/meta_data/class_index.csv'
        df = pd.read_csv(file_path)
        idx_to_labels = {}
        for idx, row in df.iterrows():
            if self.language == 'en':
                idx_to_labels[row['ID']] = [row['wordnet'], row['class']]
            else:
                idx_to_labels[row['ID']] = [row['wordnet'], row['Chinese']]
        return idx_to_labels

    def annotate_image(self,
                       img,
                       pred_softmax,
                       img_path="",
                       n=10,
                       file_path=None,
                       save_plot=True):
        if img is None:
            if self.language == 'en':
                # use opencv to load the image
                img = cv2.imread(img_path)
            else:
                img = Image.open(img_path)

        if self.language != 'en':
            draw = ImageDraw.Draw(img)

        pred_ids, confs = self.get_top_n(pred_softmax, n)
        idx_to_labels = self.get_idx_labels()

        for i in range(n):
            class_name = idx_to_labels[pred_ids[i]][1]  # get name of the category
            confidence = confs[i] * 100  # get probability of the category
            text = '{:<15} {:>.4f}'.format(class_name, confidence)
            # Annotate image on its left upper corner with predictions
            # params: location，font，size，bgr color，line width
            if self.language == 'en':
                img = cv2.putText(img, text,
                                  (25, 50 + 40 * i),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  1.25,
                                  (0, 0, 255),
                                  3)
                if save_plot:
                    _ = cv2.imwrite('output/img_pred.jpg', img)
            else:
                draw.text((50, 100 + 50 * i), text, font=font, fill=(255, 0, 0, 1))
                if save_plot:
                    img.save('output/img_pred.jpg')
        # return Image.open('output/img_pred.jpg')
        return img

    def pair_plot(self, img_path, img_pred, pred_softmax):
        fig = plt.figure(figsize=(18, 6))
        # Plot the annonated figure on the right hand-side
        ax1 = plt.subplot(1, 2, 1)
        if self.language == 'en':
            img_rgb = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)  # BGR转RGB
            img_pil = Image.fromarray(img_rgb)  # array 转 PIL
            ax1.imshow(img_pil)
        ax1.imshow(img_pred)
        ax1.axis('off')

        # Plot bar chart for the probability of different classes
        ax2 = plt.subplot(1, 2, 2)
        file_path = 'data/image_net/meta_data/class_index.csv'
        df = pd.read_csv(file_path)
        x = df['ID']
        y = pred_softmax.cpu().detach().numpy()[0]
        ax2.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)

        plt.ylim([0, 1.0])  # y axis range
        if self.language == 'en':
            plt.title('{} image classification'.format(img_path), fontsize=15, fontname='Arial')
            plt.xlabel('Class', fontsize=10, fontname='Arial')
            plt.ylabel('Confidence', fontsize=10, fontname='Arial')
        else:
            plt.title('{} 图像分类'.format(img_path), fontsize=15)
            plt.xlabel('类别', fontsize=10)
            plt.ylabel('置信度', fontsize=10)
        plt.tick_params(labelsize=10)
        plt.show()
        plt.tight_layout()
        fig.savefig('output/annotated+prob_bar.jpg')

    def prediction(self, pred_softmax, n=10):
        pred_df = pd.DataFrame()
        idx_to_labels = self.get_idx_labels()
        pred_ids, confs = self.get_top_n(pred_softmax, n)
        for i in range(n):
            class_name = idx_to_labels[pred_ids[i]][1]  # Get the class name based on ID
            label_idx = int(pred_ids[i])  # Get the class ID
            wordnet = idx_to_labels[pred_ids[i]][0]  # Get WordNet
            confidence = confs[i] * 100  # get probability
            temp = pd.DataFrame([{'Class': class_name, 'Class_ID': label_idx,
                                'Confidence(%)': confidence, 'WordNet': wordnet}])

            pred_df = pd.concat([pred_df, temp])
        return pred_df


class InferVideo(InferImage):
    def __init__(self, model) -> None:
        super().__init__(model=model)

    def pred_single_frame(self, img, n=5):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
        img_pil = Image.fromarray(img_rgb)  # array 转 pil
        input_img = self.preprocess(img_pil,
                                    show_plot=False,
                                    verbose=False)

        pred_softmax = self.inference(input_img)
        if self.language == 'en':
            img_bgr = img
            annotated_img = self.annotate_image(img=img_bgr,
                                                pred_softmax=pred_softmax,
                                                n=n,
                                                save_plot=False)
        else:
            annotated_img = self.annotate_image(img=img_pil,
                                                pred_softmax=pred_softmax,
                                                n=n,
                                                save_plot=False)
            annotated_img = cv2.cvtColor(np.array(annotated_img), cv2.COLOR_RGB2BGR)  # RGB转BGR

        return annotated_img, pred_softmax

    def pred_single_frame_bar(self, img, pred_softmax, temp_out_dir, frame_id):
        '''
        输入pred_single_frame函数输出的bgr-array,加柱状图,保存
        '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
        fig = plt.figure(figsize=(18, 6))
        # 绘制左图-视频图
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(img)
        ax1.axis('off')
        # 绘制右图-柱状图
        ax2 = plt.subplot(1, 2, 2)
        x = range(1000)
        y = pred_softmax.cpu().detach().numpy()[0]
        ax2.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
        plt.xlabel('类别', fontsize=20)
        plt.ylabel('置信度', fontsize=20)
        ax2.tick_params(labelsize=16)  # 坐标文字大小
        plt.ylim([0, 1.0])  # y轴取值范围
        plt.xlabel('类别', fontsize=25)
        plt.ylabel('置信度', fontsize=25)
        plt.title('图像分类预测结果', fontsize=30)

        plt.tight_layout()
        fig.savefig(f'{temp_out_dir}/{frame_id:06d}.jpg')
        # 释放内存
        fig.clf()
        plt.close()
        gc.collect()

    def predict_and_annotate(self, input_path, output_path='output/output_pred.mp4', with_bar=False):

        temp_out_dir = time.strftime('%Y%m%d%H%M%S')
        os.mkdir(temp_out_dir)
        print('创建文件夹 {} 用于存放每帧预测结果'.format(temp_out_dir))

        # 读入待预测视频
        imgs = mmcv.VideoReader(input_path)

        prog_bar = mmcv.ProgressBar(len(imgs))

        # 对视频逐帧处理
        for frame_id, img in enumerate(imgs):

            # 处理单帧画面
            annotated_img, pred_softmax = self.pred_single_frame(img, n=5)
            if not with_bar:
                # 将处理后的该帧画面图像文件，保存至 /tmp 目录下
                cv2.imwrite(f'{temp_out_dir}/{frame_id:06d}.jpg', annotated_img)
            else:
                img = self.pred_single_frame_bar(annotated_img, pred_softmax, temp_out_dir, frame_id)

            prog_bar.update()  # 更新进度条

        # 把每一帧串成视频文件
        mmcv.frames2video(temp_out_dir, output_path, fps=imgs.fps, fourcc='mp4v')

        shutil.rmtree(temp_out_dir)  # 删除存放每帧画面的临时文件夹
        print('删除临时文件夹', temp_out_dir)


class InferStream(InferImage):
    def __init__(self, model) -> None:
        super().__init__(model=model)

    def process_frame(self, img,n=5):
        '''
        输入摄像头拍摄画面bgr-array,输出图像分类预测结果bgr-array
        '''

        # 记录该帧开始处理的时间
        start_time = time.time()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        input_img = self.preprocess(img_pil)
        pred_softmax = self.inference(input_img)

        if self.language == 'en':
            img_bgr = img
            img = self.annotate_image(img=img_bgr,
                                                pred_softmax=pred_softmax,
                                                n=n,
                                                save_plot=False)
        else:
            img = self.annotate_image(img=img_pil,
                                                pred_softmax=pred_softmax,
                                                n=n,
                                                save_plot=False)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # RGB转BGR

        # 记录该帧处理完毕的时间
        end_time = time.time()
        # 计算每秒处理图像帧数FPS
        FPS = 1/(end_time - start_time)
        # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，线宽，线型
        img = cv2.putText(
            img, 'FPS  ' + str(int(FPS)),
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255),
            4, cv2.LINE_AA)

        return img

    def predict_and_annotate(self):
        # 获取摄像头，传入0表示获取系统默认摄像头
        cap = cv2.VideoCapture(1)

        # 打开cap
        cap.open(0)

        # 无限循环，直到break被触发
        while cap.isOpened():
            # 获取画面
            success, frame = cap.read()
            if not success:
                print('Error')
                break

            # !!!处理帧函数
            frame = self.process_frame(frame)

            # 展示处理后的三通道图像
            cv2.imshow('my_window', frame)
            plt.close()
            gc.collect()

            if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
                break

        # 关闭摄像头
        cap.release()

        # 关闭图像窗口
        cv2.destroyAllWindows()
