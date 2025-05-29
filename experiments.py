import tensorflow as tf
import os
import utils
from Models import TransformersV2, CNN_RNN, PopularModels
from tools import callbacks
import numpy as np
from tools import metrics
from tools.metrics import print_and_save_results
import tensorflow.keras.backend as K
from scipy.signal import resample
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt


class Experiment():
    def __init__(self, base_project_dir='.', seed=0, network_structure='Transformer', exp_name = '10fold-SIT-32channel'):
        self.base_project_dir = base_project_dir
        self.seed = seed
        self.network_structure = network_structure
        self.exp_name = exp_name
        self. task = str()
        self.electrode_names = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'FC5', 'FC1', 'FC2', 'FC6',
            'T7', 'C3', 'Cz', 'C4', 'T8',
            'CP5', 'CP1', 'CP2', 'CP6',
            'P7', 'P3', 'Pz', 'P4', 'P8',
            'O1', 'Oz', 'O2',
            'AFz', 'FCz', 'CPz', 'POz'
        ]


        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
        tf.random.set_seed(seed)


    def prepare(self):

        HC_data, PD_OFF_data, PD_ON_data = utils.load_dataset_UcSanDiego(self.base_project_dir)
        HC_data_UI, HC_labels_UI, PD_data_UI, PD_labels_UI = utils.load_dataset_UI(self.base_project_dir)

        ########################### UcSanDiego ###########################
        HC_labels = np.zeros((HC_data.shape[0], 1))
        PD_OFF_labels = np.ones((PD_OFF_data.shape[0], 1))

        self.X = np.vstack((HC_data, PD_OFF_data))
        self.Y = np.vstack((HC_labels, PD_OFF_labels))

        self.X = self.X[:, :32, :]
        if self.exp_name.split('-')[-1] == '31channel':
            self.X = np.delete(self.X, 12, 1)
        ##################################################################


        ############################### UI ###############################
        self.X_UI = np.vstack((HC_data_UI, PD_data_UI))
        self.Y_UI = np.vstack((HC_labels_UI, PD_labels_UI))

        '''Setting the 64th channel equal to zero. In the next step, when changing channel orders, it will be placed as the 13th channel (or of Pz channel)'''
        X_temp = np.zeros((self.X_UI.shape[0], 64, self.X_UI.shape[2]))
        X_temp[:, :63, :] = self.X_UI
        self.X_UI = X_temp

        '''Channel selection for maching with other datasets'''    #Pz
        channel_numbers = [1, 33, 4, 3, 7, 6, 9, 8, 12, 11, 14, 13, 64, 46, 15, 16, 17, 48, 18, 19, 21, 22, 24, 25, 27, 28, 29, 30, 61, 31, 2, 23]                                                                
        channel_indxs = [(i-1) for i in channel_numbers]
        self.X_UI = self.X_UI[:, channel_indxs, :]

        if self.exp_name.split('-')[-1] == '31channel':
            self.X_UI = np.delete(self.X_UI, 12, 1)
        ##################################################################

        if self.exp_name.split('-')[-1] == '31channel' and self.network_structure != 'Transformer':
            self.X = self.X.reshape((-1, 512, 31))
            self.X_UI = self.X_UI.reshape((-1, 512, 31))
        elif  self.exp_name.split('-')[-1] == '32channel' and self.network_structure != 'Transformer':
            self.X = self.X.reshape((-1, 512, 32))
            self.X_UI = self.X_UI.reshape((-1, 512, 32))

        self.input_shape = self.X.shape[1:]
            
        print('Input Shape:', self.input_shape)


    def train_10fold(self):
        
        if self.network_structure == 'Transformer':
            self.classifier = TransformersV2.transfomerBasedModel(self.input_shape)
        elif self.network_structure == 'VGG_BiLSTM_Attn':
            self.classifier = CNN_RNN.VGG_BiLSTM_Attn_Model(self.input_shape)
        elif self.network_structure == 'VGG16':
            self.classifier = PopularModels.VGG16_Model(self.input_shape)
        elif self.network_structure == 'ResNet18':
            self.classifier = PopularModels.ResNet18_Model(self.input_shape)
        elif self.network_structure == 'EfficientNet':
            self.classifier = PopularModels.EfficientNet_Model(self.input_shape)
        elif self.network_structure == 'MobileNet':
            self.classifier = PopularModels.MobileNet_Model(self.input_shape)
        elif self.network_structure == 'Inceptionv3':
            self.classifier = PopularModels.Inceptionv3_Model(self.input_shape)

        self.task = self.task + self.network_structure
        self.task = self.task + '-' + self.exp_name

        early_stoping_clb = tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_loss', restore_best_weights=True)
        thresholding_clb = callbacks.Thresholding('val_accuracy', 0.96)

        opt = tf.keras.optimizers.Adam(learning_rate=2e-6, clipvalue=5.0)

        self.classifier.summary()
        
        self.classifier.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics='accuracy'
        )

        self.results_dir = os.path.join(self.base_project_dir, 'Results')
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

        self.results_dir = os.path.join(self.results_dir, self.task)
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        
        TrainedModels_dir = os.path.join(self.base_project_dir, 'TrainedModels')
        if not os.path.exists(TrainedModels_dir):
            os.mkdir(TrainedModels_dir)

        TrainedModels_dir = os.path.join(TrainedModels_dir, self.task)
        if not os.path.exists(TrainedModels_dir):
            os.mkdir(TrainedModels_dir)

        print('Task:', self.task)

        self.classifier.save_weights('initial_weight.h5')

        Y_test_set = list()
        Y_pred_set = list()

        ten_fold = utils.Dataset10FoldSpliter(self.X, self.Y, shuffle=True, seed=self.seed)
        epoches_list = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

        for i in range(10):
            self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test, self.test_tracker = ten_fold.split()

            print('Training ' + str(i+1) + 'th model. Train HC class: ' + str(self.Y_train.shape[0] - np.sum(self.Y_train)) + ' - Test HC class: ' + str(self.Y_test.shape[0] - np.sum(self.Y_test)))
            tf.keras.backend.clear_session()
            self.classifier.load_weights("initial_weight.h5")
            
            try :
                self.classifier.fit(
                    self.X_train,
                    self.Y_train,
                    #epochs=100,
                    epochs=epoches_list[i],
                    batch_size=32,
                    #callbacks=[thresholding_clb, ],
                    validation_data=(self.X_val, self.Y_val),
                )
            except KeyboardInterrupt:
                pass
            
            Y_pred = self.classifier.predict(self.X_val)

            Y_test_set.append(self.Y_val)
            Y_pred_set.append(Y_pred)

            save_results_to = os.path.join(self.results_dir, self.task + '_fold' + str(i+1) + '.txt')
            print_and_save_results(self.Y_val, Y_pred, save_to=save_results_to)

            output_moedel_path = os.path.join(TrainedModels_dir, self.task + str(i+1) + '.h5')
            self.classifier.save_weights(output_moedel_path)

        print(50*"=", 'Final Results on 10-Fold', 50*"=")
        self.evaluate(Y_test_set, Y_pred_set)
        os.remove("initial_weight.h5")
    

    def reproduce_results_on_10fold(self, evaluation_set='test', plot_attention_weights=False, plot_self_attention_weights=False):
        if self.network_structure == 'Transformer':
            self.classifier = TransformersV2.transfomerBasedModel(self.input_shape)
        elif self.network_structure == 'VGG_BiLSTM_Attn':
            self.classifier = CNN_RNN.VGG_BiLSTM_Attn_Model(self.input_shape)
        elif self.network_structure == 'VGG16':
            self.classifier = PopularModels.VGG16_Model(self.input_shape)
        elif self.network_structure == 'ResNet18':
            self.classifier = PopularModels.ResNet18_Model(self.input_shape)
        elif self.network_structure == 'EfficientNet':
            self.classifier = PopularModels.EfficientNet_Model(self.input_shape)
        elif self.network_structure == 'MobileNet':
            self.classifier = PopularModels.MobileNet_Model(self.input_shape)
        elif self.network_structure == 'Inceptionv3':
            self.classifier = PopularModels.Inceptionv3_Model(self.input_shape)

        self.task = self.task + self.network_structure
        self.task = self.task + '-' + self.exp_name

        self.classifier.summary()

        self.results_dir = os.path.join(self.base_project_dir, 'Results')
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

        self.results_dir = os.path.join(self.results_dir, self.task)
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

        TrainedModels_dir = os.path.join(self.base_project_dir, 'TrainedModels')
        if not os.path.exists(TrainedModels_dir):
            os.mkdir(TrainedModels_dir)

        TrainedModels_dir = os.path.join(TrainedModels_dir, self.task)
        if not os.path.exists(TrainedModels_dir):
            os.mkdir(TrainedModels_dir)

        print('Task:', self.task)

        Y_test_set = list()
        Y_pred_set = list()

        ten_fold = utils.Dataset10FoldSpliter(self.X, self.Y, shuffle=True, seed=self.seed)

        for i in range(10):
            self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test, self.test_tracker = ten_fold.split()

            print('Evaluating ' + str(i+1) + 'th model.')

            trained_moedel_path = os.path.join(TrainedModels_dir, self.task + str(i+1) + '.h5')
            self.classifier.load_weights(trained_moedel_path)

            if evaluation_set == 'val':
                Y_pred = self.classifier(self.X_val)
                Y_pred_set.append(Y_pred)
                Y_test_set.append(self.Y_val)
                save_results_to = os.path.join(self.results_dir, self.task + '_fold' + str(i+1) + '_val' + '.txt')
                print_and_save_results(self.Y_val, Y_pred, save_to=save_results_to)
            else:
                Y_pred = self.classifier(self.X_test)
                Y_pred_set.append(Y_pred)
                Y_test_set.append(self.Y_test)
                save_results_to = os.path.join(self.results_dir, self.task + '_fold' + str(i+1) + '_test' + '.txt')
                print_and_save_results(self.Y_test, Y_pred, save_to=save_results_to)

            ## **Plot Additive Attention Weights** ##
            if plot_attention_weights:
                sampling_rate = 256
                attention_weight_predictor = tf.keras.Model(self.classifier.inputs, self.classifier.get_layer('additive_attention').output[1])  
                attention_weight_test = attention_weight_predictor(self.X_test).numpy()
                self.X_test = np.transpose(self.X_test, (0, 2, 1))
                for test_eeg_idx in range(self.X_test.shape[0]):
                    attention_weight = resample(attention_weight_test[test_eeg_idx, :, 0], self.X_test.shape[1])[:self.X_test.shape[1]]
                    save_path = os.path.join(self.results_dir, f'Additive_attention_weights_fold{i+1}_test_data{test_eeg_idx+1}.png')
                    label = 'PD' if self.Y_test[test_eeg_idx] == 1 else 'HC'
                    utils.plot_32_channel_eeg_with_background_attention(self.X_test[test_eeg_idx], attention_weight, self.electrode_names, sampling_rate, label, save_path)
            
            ## **Plot Self-Attention Weights from Transformer** ##
            if plot_self_attention_weights:
                print(f"Extracting self-attention weights from Transformer Encoder for fold {i+1}...")

                # Extract attention maps from the encoder
                attention_layer = tf.keras.Model(self.classifier.inputs, self.classifier.get_layer('encoder').output[1])  
                attention_map_h1, attention_map_h2 = attention_layer(self.X_test)  # Extract attention maps

                # Compute mean attention maps across all test samples
                mean_attention_h1 = np.mean(attention_map_h1, axis=0)
                mean_attention_h2 = np.mean(attention_map_h2, axis=0)
                mean_attention_total = (mean_attention_h1 + mean_attention_h2) / 2  # Mean across both heads

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # Define font sizes
                label_fontsize = 12
                tick_fontsize = 8
                title_fontsize = 14

                # Plot Head 1 attention map
                im1 = axes[0].imshow(mean_attention_h1, cmap='viridis', interpolation='nearest')
                axes[0].set_title(f'Scale-dot-product Attentio Map - Head 1', fontsize=title_fontsize)
                #axes[0].set_xlabel('Key Positions', fontsize=label_fontsize)
                #axes[0].set_ylabel('Query Positions', fontsize=label_fontsize)
                axes[0].set_xticks(np.arange(len(self.electrode_names)))
                axes[0].set_yticks(np.arange(len(self.electrode_names)))
                axes[0].set_xticklabels(self.electrode_names, rotation=90, fontsize=tick_fontsize)
                axes[0].set_yticklabels(self.electrode_names, fontsize=tick_fontsize)

                # Plot Head 2 attention map
                im2 = axes[1].imshow(mean_attention_h2, cmap='viridis', interpolation='nearest')
                axes[1].set_title(f'Scale-dot-product Attentio Map - Head 1', fontsize=title_fontsize)
                #axes[1].set_xlabel('Key Positions', fontsize=label_fontsize)
                #axes[1].set_ylabel('Query Positions', fontsize=label_fontsize)
                axes[1].set_xticks(np.arange(len(self.electrode_names)))
                axes[1].set_yticks(np.arange(len(self.electrode_names)))
                axes[1].set_xticklabels(self.electrode_names, rotation=90, fontsize=tick_fontsize)
                axes[1].set_yticklabels(self.electrode_names, fontsize=tick_fontsize)

                # Plot the Mean Attention Map (across both heads)
                im3 = axes[2].imshow(mean_attention_total, cmap='magma', interpolation='nearest')
                axes[2].set_title(f'Mean Scale-dot-product Attentio Map', fontsize=title_fontsize)
                #axes[2].set_xlabel('Key Positions', fontsize=label_fontsize)
                #axes[2].set_ylabel('Query Positions', fontsize=label_fontsize)
                axes[2].set_xticks(np.arange(len(self.electrode_names)))
                axes[2].set_yticks(np.arange(len(self.electrode_names)))
                axes[2].set_xticklabels(self.electrode_names, rotation=90, fontsize=tick_fontsize)
                axes[2].set_yticklabels(self.electrode_names, fontsize=tick_fontsize)

                # Add colorbars
                fig.colorbar(im1, ax=axes[0])
                fig.colorbar(im2, ax=axes[1])
                fig.colorbar(im3, ax=axes[2])

                plt.tight_layout()

                # Save figure instead of showing it
                save_path = os.path.join(self.results_dir, f'self_attention_fold_{i+1}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved self-attention map for fold {i+1} to: {save_path}")



        print(50*"=", 'Final Results on 10-Fold', 50*"=")
        self.evaluate(Y_test_set, Y_pred_set)



    def train_10fold_SIT(self):
        if self.exp_name.split('-')[-1] == '31channel':
            input_shape = (31, 512)
        else:
            input_shape = (32, 512)


        if self.network_structure == 'Transformer':
            self.classifier = TransformersV2.transfomerBasedModel(self.input_shape)
        elif self.network_structure == 'VGG_BiLSTM_Attn':
            self.classifier = CNN_RNN.VGG_BiLSTM_Attn_Model(self.input_shape)
        elif self.network_structure == 'VGG16':
            self.classifier = PopularModels.VGG16_Model(self.input_shape)
        elif self.network_structure == 'ResNet18':
            self.classifier = PopularModels.ResNet18_Model(self.input_shape)
        elif self.network_structure == 'EfficientNet':
            self.classifier = PopularModels.EfficientNet_Model(self.input_shape)
        elif self.network_structure == 'MobileNet':
            self.classifier = PopularModels.MobileNet_Model(self.input_shape)
        elif self.network_structure == 'Inceptionv3':
            self.classifier = PopularModels.Inceptionv3_Model(self.input_shape)


        early_stoping_clb = tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_accuracy', mode='max', restore_best_weights=True)
        thresholding_clb = callbacks.Thresholding('val_loss', 0.0)

        opt = tf.keras.optimizers.Adam(learning_rate=5e-6, clipvalue=5.0)

        self.classifier.summary()
        
        self.classifier.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics='accuracy'
        )

        self.results_dir = os.path.join(self.base_project_dir, 'Results')
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

        self.results_dir = os.path.join(self.results_dir, self.task)
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        
        TrainedModels_dir = os.path.join(self.base_project_dir, 'TrainedModels')
        if not os.path.exists(TrainedModels_dir):
            os.mkdir(TrainedModels_dir)

        TrainedModels_dir = os.path.join(TrainedModels_dir, self.task)
        if not os.path.exists(TrainedModels_dir):
            os.mkdir(TrainedModels_dir)

        print('Task:', self.task)

        self.classifier.save_weights('initial_weight.h5')

        Y_test_set = list()
        Y_pred_set = list()

        HC_names, PD_OFF_names = utils.get_UcSanDiego_subject_names(self.base_project_dir)
        subject_names = HC_names + PD_OFF_names
        ten_fold = utils.SubjectNames10FoldSpliter(subject_names, shuffle=True, seed=self.seed)
        epoches_list = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        for i in range(10):
            subject_names_train, subject_names_val, subject_names_test, _ = ten_fold.split()
            self.X_train, self.Y_train = utils.load_UcSanDiego_based_on_subject_names(subject_names_train, self.base_project_dir)
            self.X_val, self.Y_val = utils.load_UcSanDiego_based_on_subject_names(subject_names_val, self.base_project_dir)
            self.X_test, self.Y_test = utils.load_UcSanDiego_based_on_subject_names(subject_names_test, self.base_project_dir)
            
            self.X_train = self.X_train[:, :32, :]
            self.X_val = self.X_val[:, :32, :]
            self.X_test = self.X_test[:, :32, :]

            if self.exp_name.split('-')[-1] == '31channel':
                self.X_train = np.delete(self.X_train, 12, 1)
                self.X_val = np.delete(self.X_val, 12, 1)
                self.X_test = np.delete(self.X_test, 12, 1)


            print('Training ' + str(i+1) + 'th model. Train HC class: ' + str(self.Y_train.shape[0] - np.sum(self.Y_train)) + ' - Test HC class: ' + str(self.Y_test.shape[0] - np.sum(self.Y_test)))
            tf.keras.backend.clear_session()
            self.classifier.load_weights("initial_weight.h5")
            
            try :
                self.classifier.fit(
                    self.X_train,
                    self.Y_train,
                    epochs=epoches_list[i],
                    batch_size=64,
                    callbacks=[early_stoping_clb, ],
                    validation_data=(self.X_val, self.Y_val),
                )
            except KeyboardInterrupt:
                pass
            
            Y_pred = self.classifier.predict(self.X_val)

            Y_test_set.append(self.Y_val)
            Y_pred_set.append(Y_pred)

            save_results_to = os.path.join(self.results_dir, self.task + '_fold' + str(i+1) + '.txt')
            print_and_save_results(self.Y_val, Y_pred, save_to=save_results_to)

            output_moedel_path = os.path.join(TrainedModels_dir, self.task + str(i+1) + '.h5')
            self.classifier.save_weights(output_moedel_path)

        print(50*"=", 'Final Results on 10-Fold', 50*"=")
        self.evaluate(Y_test_set, Y_pred_set)
        os.remove("initial_weight.h5")


    def reproduce_results_on_10fold_SIT(self, evaluation_set='test', plot_attention_weights=False):
        if self.exp_name.split('-')[-1] == '31channel':
            input_shape = (31, 512)
        else:
            input_shape = (32, 512)
        if self.network_structure == 'Transformer':
            self.classifier = TransformersV2.transfomerBasedModel(self.input_shape)
        elif self.network_structure == 'VGG_BiLSTM_Attn':
            self.classifier = CNN_RNN.VGG_BiLSTM_Attn_Model(self.input_shape)
        elif self.network_structure == 'VGG16':
            self.classifier = PopularModels.VGG16_Model(self.input_shape)
        elif self.network_structure == 'ResNet18':
            self.classifier = PopularModels.ResNet18_Model(self.input_shape)
        elif self.network_structure == 'EfficientNet':
            self.classifier = PopularModels.EfficientNet_Model(self.input_shape)
        elif self.network_structure == 'MobileNet':
            self.classifier = PopularModels.MobileNet_Model(self.input_shape)
        elif self.network_structure == 'Inceptionv3':
            self.classifier = PopularModels.Inceptionv3_Model(self.input_shape)

        self.task = self.task + self.network_structure
        self.task = self.task + '-' + self.exp_name

        self.classifier.summary()

        self.results_dir = os.path.join(self.base_project_dir, 'Results')
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

        self.results_dir = os.path.join(self.results_dir, self.task)
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        
        TrainedModels_dir = os.path.join(self.base_project_dir, 'TrainedModels')
        if not os.path.exists(TrainedModels_dir):
            os.mkdir(TrainedModels_dir)

        TrainedModels_dir = os.path.join(TrainedModels_dir, self.task)
        if not os.path.exists(TrainedModels_dir):
            os.mkdir(TrainedModels_dir)

        print('Task:', self.task)

        Y_test_set = list()
        Y_pred_set = list()

        HC_names, PD_OFF_names = utils.get_UcSanDiego_subject_names(self.base_project_dir)
        subject_names = HC_names + PD_OFF_names
        ten_fold = utils.SubjectNames10FoldSpliter(subject_names, shuffle=True, seed=self.seed)

        for i in range(10):
            subject_names_train, subject_names_val, subject_names_test, _ = ten_fold.split()
            self.X_train, self.Y_train = utils.load_UcSanDiego_based_on_subject_names(subject_names_train, self.base_project_dir)
            self.X_val, self.Y_val = utils.load_UcSanDiego_based_on_subject_names(subject_names_val, self.base_project_dir)
            self.X_test, self.Y_test = utils.load_UcSanDiego_based_on_subject_names(subject_names_test, self.base_project_dir)
            
            self.X_train = self.X_train[:, :32, :]
            self.X_val = self.X_val[:, :32, :]
            self.X_test = self.X_test[:, :32, :]

            if self.exp_name.split('-')[-1] == '31channel':
                self.X_train = np.delete(self.X_train, 12, 1)
                self.X_val = np.delete(self.X_val, 12, 1)
                self.X_test = np.delete(self.X_test, 12, 1)


            print('Evaluating ' + str(i+1) + 'th model.')

            trained_moedel_path = os.path.join(TrainedModels_dir, self.task + str(i+1) + '.h5')
            self.classifier.load_weights(trained_moedel_path)

            if evaluation_set == 'val':
                Y_pred = self.classifier(self.X_val)
                Y_pred_set.append(Y_pred)
                Y_test_set.append(self.Y_val)
                save_results_to = os.path.join(self.results_dir, self.task + '_fold' + str(i+1) + '_val' + '.txt')
                print_and_save_results(self.Y_val, Y_pred, save_to=save_results_to)
            else:
                Y_pred = self.classifier(self.X_test)
                Y_pred_set.append(Y_pred)
                Y_test_set.append(self.Y_test)
                save_results_to = os.path.join(self.results_dir, self.task + '_fold' + str(i+1) + '_test' + '.txt')
                print_and_save_results(self.Y_test, Y_pred, save_to=save_results_to)


            if plot_attention_weights:
                use_channel = 0
                sampling_rate = 50
                attention_weight_predictor = tf.keras.Model(self.classifier.inputs, self.classifier.get_layer('additive_attention').output[1])  
                attention_weight_test = attention_weight_predictor(self.X_test).numpy()
                self.X_test = np.transpose(self.X_test, (0, 2, 1))
                for test_eeg_idx in range(self.X_test.shape[0]):
                    attention_weight = resample(attention_weight_test[test_eeg_idx, :, use_channel], self.X_test.shape[1])[:self.X_test.shape[1]]
                    attention_weight = attention_weight * 100
                    time_axies = list(map(lambda x:x/sampling_rate, [i for i in range(self.X_test.shape[1])]))
                    plt.figure()
                    plt.plot(time_axies, self.X_test[test_eeg_idx, :, use_channel], linewidth=2, color='black')
                    plt.fill_between(time_axies, attention_weight, step="pre", color='red', alpha=0.2)
                    plt.title('EEG Signal with Attention Weights')
                    plt.legend(['EEG Signal', 'Attention Weights'])
                    plt.show()
                    if input('Enter \'q\' for pass the attention weight plots.') == 'q':
                        break
                 
        print(50*"=", 'Final Results on 10-Fold', 50*"=")
        self.evaluate(Y_test_set, Y_pred_set)




    def evaluate(self, Y_test_set, Y_pred_set, test=False):
        Y_pred_total = Y_pred_set[0]
        Y_total = Y_test_set[0]
        for i in range(1, len(Y_pred_set)):
            Y_pred_total = np.vstack((Y_pred_total, Y_pred_set[i]))
            Y_total = np.vstack((Y_total, Y_test_set[i]))

        if test:
            save_results_to = os.path.join(self.results_dir, self.task+ '_test' + '.txt')
        else:
            save_results_to = os.path.join(self.results_dir, self.task + '.txt')

        print_and_save_results(Y_total, Y_pred_total, save_to=save_results_to)


    def evalute_on_PRED_CT(self, model_number):
        PD_OFF_data, _ = utils.load_dataset_PRED_CT(self.base_project_dir)

        eeg_data = PD_OFF_data
        labels = np.ones((eeg_data.shape[0], 1))

        '''Channel selection for maching with other datasets'''
        channel_numbers = [1, 34, 4, 3, 7, 6, 9, 8, 12, 11, 15, 14, 13, 48, 16, 17, 18, 50, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 62, 32, 2, 24]   # original
        channel_indxs = [(i-1) for i in channel_numbers]
        eeg_data = eeg_data[:, channel_indxs, :]

        if self.exp_name.split('-')[-1] == '31channel':
            eeg_data = np.delete(eeg_data, 12, 1)

        self.input_shape = eeg_data.shape[1:]

        if self.network_structure == 'Transformer':
            self.classifier = TransformersV2.transfomerBasedModel(self.input_shape)
        elif self.network_structure == 'VGG_BiLSTM_Attn':
            self.classifier = CNN_RNN.VGG_BiLSTM_Attn_Model(self.input_shape)
        elif self.network_structure == 'VGG16':
            self.classifier = PopularModels.VGG16_Model(self.input_shape)
        elif self.network_structure == 'ResNet18':
            self.classifier = PopularModels.ResNet18_Model(self.input_shape)
        elif self.network_structure == 'EfficientNet':
            self.classifier = PopularModels.EfficientNet_Model(self.input_shape)
        elif self.network_structure == 'MobileNet':
            self.classifier = PopularModels.MobileNet_Model(self.input_shape)
        elif self.network_structure == 'Inceptionv3':
            self.classifier = PopularModels.Inceptionv3_Model(self.input_shape)

        self.classifier.summary()
        self.task = self.task + self.network_structure
        self.task = self.task + '-' + self.exp_name

        TrainedModels_dir = os.path.join(self.base_project_dir, 'TrainedModels')
        if not os.path.exists(TrainedModels_dir):
            os.mkdir(TrainedModels_dir)

        TrainedModels_dir = os.path.join(TrainedModels_dir, self.task)
        if not os.path.exists(TrainedModels_dir):
            os.mkdir(TrainedModels_dir)

        trained_moedel_path = os.path.join(TrainedModels_dir, self.task + str(model_number) + '.h5')

        self.classifier.load_weights(trained_moedel_path)


        '''Dviding data into several group because the number of segments are too high to predict the labels all at once '''
        n_full_groups = eeg_data.shape[0] // 1000
        Y_pred_list = []
        for i in range (n_full_groups+1):
            if i < n_full_groups:
                Y_pred_list.append(self.classifier(eeg_data[1000*i:1000*(i+1)]))
            elif i == n_full_groups:
                Y_pred_list.append(self.classifier(eeg_data[1000*i:]))
        
        Y_pred = Y_pred_list[0]
        for i in range(1, len(Y_pred_list)):
            Y_pred = np.vstack((Y_pred, Y_pred_list[i]))

        #start_range = 2000
        #end_range = 2100
        #print(np.hstack((labels[start_range:end_range], Y_pred[start_range:end_range])))

        print('\n')
        print(30*'=', 'Results', 30*'=')
        print('Accuracy:', float(metrics.accuracy(labels, Y_pred)))




    def evalute_on_UI(self, model_number):
        HC_data, HC_labels, PD_data, PD_labels = utils.load_dataset_UI(self.base_project_dir)

        self.X = np.vstack((HC_data, PD_data))
        self.Y = np.vstack((HC_labels, PD_labels))


        '''Setting the 64th channel equal to zero. In the next step, when changing channel orders, it will be placed as the 13th channel instead of Pz'''
        X_temp = np.zeros((self.X.shape[0], 64, self.X.shape[2]))
        X_temp[:, :63, :] = self.X
        self.X = X_temp

        '''Channel selection for maching with other datasets'''    #Pz
        channel_numbers = [1, 33, 4, 3, 7, 6, 9, 8, 12, 11, 14, 13, 64, 46, 15, 16, 17, 48, 18, 19, 21, 22, 24, 25, 27, 28, 29, 30, 61, 31, 2, 23]                                                                
        channel_indxs = [(i-1) for i in channel_numbers]
        self.X = self.X[:, channel_indxs, :]

        if self.exp_name.split('-')[-1] == '31channel':
            self.X = np.delete(self.X, 12, 1)

        self.input_shape = self.X.shape[1:]

        if self.network_structure == 'Transformer':
            self.classifier = TransformersV2.transfomerBasedModel(self.input_shape)
        elif self.network_structure == 'VGG_BiLSTM_Attn':
            self.classifier = CNN_RNN.VGG_BiLSTM_Attn_Model(self.input_shape)
        elif self.network_structure == 'VGG16':
            self.classifier = PopularModels.VGG16_Model(self.input_shape)
        elif self.network_structure == 'ResNet18':
            self.classifier = PopularModels.ResNet18_Model(self.input_shape)
        elif self.network_structure == 'EfficientNet':
            self.classifier = PopularModels.EfficientNet_Model(self.input_shape)
        elif self.network_structure == 'MobileNet':
            self.classifier = PopularModels.MobileNet_Model(self.input_shape)
        elif self.network_structure == 'Inceptionv3':
            self.classifier = PopularModels.Inceptionv3_Model(self.input_shape)

        self.classifier.summary()
        self.task = self.task + self.network_structure
        self.task = self.task + '-' + self.exp_name

        TrainedModels_dir = os.path.join(self.base_project_dir, 'TrainedModels')
        if not os.path.exists(TrainedModels_dir):
            os.mkdir(TrainedModels_dir)

        TrainedModels_dir = os.path.join(TrainedModels_dir, self.task)
        if not os.path.exists(TrainedModels_dir):
            os.mkdir(TrainedModels_dir)

        self.results_dir = os.path.join(self.base_project_dir, 'Results')
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

        self.results_dir = os.path.join(self.results_dir, self.task)
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

        trained_moedel_path = os.path.join(TrainedModels_dir, self.task + str(model_number) + '.h5')

        self.classifier.load_weights(trained_moedel_path)

        Y_pred = self.classifier.predict(self.X)

        permutation = np.random.permutation(self.X.shape[0])
        self.Y = self.Y[permutation].reshape(-1, 1)
        Y_pred = Y_pred[permutation].reshape(-1, 1)

        print(np.hstack((self.Y[:30], Y_pred[:30])))

        save_results_to = os.path.join(self.results_dir, self.task + '_UI.txt')
        print_and_save_results(self.Y, Y_pred, save_to=save_results_to)

        '''
        Dviding data into several group because the number of segments are too high to predict the labels all at once 
        n_full_groups = self.X.shape[0] // 1000
        Y_pred_list = []
        for i in range (n_full_groups+1):
            if i < n_full_groups:
                Y_pred_list.append(self.classifier(self.X[1000*i:1000*(i+1)]))
            elif i == n_full_groups:
                Y_pred_list.append(self.classifier(self.X[1000*i:]))
        
        Y_pred = Y_pred_list[0]
        for i in range(1, len(Y_pred_list)):
            Y_pred = np.vstack((Y_pred, Y_pred_list[i]))
        '''
        










