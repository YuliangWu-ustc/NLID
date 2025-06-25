# FCN model
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 
import os

from utils.utils import save_logs
from utils.utils import calculate_metrics

# 设置环境变量，指定使用GPU 0和1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class Classifier_FCN:
    """
    实现一个全卷积网络(FCN)分类器，用于时间序列分类任务
    
    Attributes:
        output_directory (str): 模型输出目录路径
        input_shape (tuple): 输入数据的形状
        nb_classes (int): 分类类别数量
        verbose (bool): 是否显示模型摘要信息
        model (keras.Model): 构建的FCN模型实例
    """

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        """
        初始化FCN分类器
        
        Args:
            output_directory (str): 保存模型和权重的目录路径
            input_shape (tuple): 输入数据的形状(如时间步长,特征维度)
            nb_classes (int): 分类的类别数量
            verbose (bool, optional): 是否打印模型摘要. Defaults to False.
            build (bool, optional): 是否立即构建模型. Defaults to True.
        """
        self.output_directory = output_directory
        if build:
            # 使用MirroredStrategy进行多GPU训练
            self.strategy = tf.distribute.MirroredStrategy()
            with self.strategy.scope():
                self.model = self.build_model(input_shape, nb_classes)  # ⭐ 构建FCN模型结构
                if verbose:
                    self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory+'model_init.weights.h5')  # ⭐ 保存初始权重
        return
    
    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
            metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
            min_lr=0.0001)

        file_path = self.output_directory+'best_model.keras'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
            save_best_only=True)

        self.callbacks = [reduce_lr,model_checkpoint]

        return model 

    def fit(self, x_train, y_train, x_val, y_val, y_true, nb_epochs=2000):
        if not tf.test.is_gpu_available():
            print('error: no GPU available')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training  
        batch_size = 16

        mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

        start_time = time.time() 

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
            verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)
        
        duration = time.time() - start_time

        self.model.save(self.output_directory+'last_model.keras')

        model = keras.models.load_model(self.output_directory+'best_model.keras')

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer 
        y_pred = np.argmax(y_pred , axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics = True):
        model_path = self.output_directory + 'best_model.keras'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred