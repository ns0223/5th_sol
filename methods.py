from numba import jit
from numba import njit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1DTranspose
from datetime import datetime, timedelta
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from statsmodels.tsa.seasonal import STL
import numpy as np
from numpy import distutils
import distutils
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pickle


### Read Data
def save2pickle(data, path_name = 'data.pickle'):
  with open(path_name,'wb') as pickle_file:
    pickle.dump(data, pickle_file)

def read_pickle(path_name = 'data.pickle'):
  with open(path_name,'rb') as pickle_file:
    return pickle.load(pickle_file)

### Dynamic Time Warping ###
@jit
def weighted_DTW(series_1, series_2, weight):
    #### Series_1 and Series_2 are time series list, weight is weighted matrix list with two dimention
    #### default weight can be np.ones(len(series_1))
    #### weight stands for different attentions on the sub-patterns
    l1 = len(series_1)
    l2 = len(series_2)
    cum_sum = np.full((l1 + 1, l2 + 1), np.inf)
    cum_sum[0, 0] = 0.
    for i in range(l1):
        for j in range(l2):
            diff = (series_1[i] - series_2[j])
            distance = diff*diff*weight[i][j]
            cum_sum[i + 1, j + 1] = distance
            cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1],
                                            cum_sum[i + 1, j],
                                            cum_sum[i, j])
    acc_cost_mat = cum_sum[1:, 1:]
    return np.sqrt(acc_cost_mat[-1][-1])

### Refined Motif Discovery ###
### We didn't specified attention/weight and threshold in the competition for simplicity (used Adoptable threshold and flat weight)
class RM_model:
    """A class for representative daily data discovery"""

    ### fixed threshold
    def motif_dtw_24_w(self, time_series, m, threshold, weight):
        weighted_matrix = np.ones((len(weight), len(weight)))
        for i in range(len(weight)):
            for j in range(len(weight)):
                weighted_matrix[i][j] = max(weight[i], weight[j])
        # find the motif with most sub-patterns near than threshold
        length = len(time_series)-m
        sliding = int(length/m)
        distance_profile = []
        average_profile = []
        motif_co = 0
        time_series = np.array(time_series)
        for i in range(sliding+1):
            pattern = time_series[i*m:i*m+m]
            counter = 0
            average_profile.append(0)
            for j in range(sliding+1):
                distance = weighted_DTW(pattern, time_series[j*m:j*m+m], weighted_matrix)
                average_profile[i] = average_profile[i]+distance
                if distance < threshold:
                    counter+=1
            average_profile[i] = average_profile[i]/sliding
            distance_profile.append(counter - average_profile[i]*0.001)
        motif_co = np.argsort(distance_profile)[-1]
        motif_se = np.argsort(distance_profile)[-2]
        motif = dict (
            distance_profile = distance_profile,
            motif_position = motif_co*m,
            motif_sec = motif_se*m,
            average_profile = average_profile
        )

    ### Adoptable threshold
    def motif_dtw_24_w_dy(self, time_series, m, weight):
        weighted_matrix = np.ones((len(weight), len(weight)))
        for i in range(len(weight)):
            for j in range(len(weight)):
                weighted_matrix[i][j] = max(weight[i], weight[j])
        # find the motif with most sub-patterns near than threshold
        length = len(time_series)-m
        sliding = int(length/m)
        distance_stack = []
        distance_profile = []
        average_profile = []
        motif_co = 0
        # change to array for numba accelerataion
        time_series = np.array(time_series)
        for i in range(sliding+1):
            pattern = time_series[i*m:i*m+m]
            average_profile.append(0)
            for j in range(sliding+1):
                distance = weighted_DTW(pattern, time_series[j*m:j*m+m], weighted_matrix)
                average_profile[i] = average_profile[i]+distance
                distance_stack.append(distance)
            average_profile[i] = average_profile[i]/sliding
        threshold = np.median(distance_stack)
        # print('Distance matrix: ',distance_stack)
        for k in range(sliding+1):  
            pattern = time_series[i*m:i*m+m]  
            counter = 0
            for l in range(sliding+1):
                distance = distance_stack[k*(sliding+1)+l]
                if distance < threshold:
                    counter+=1
            distance_profile.append(counter - average_profile[k]*0.001)
        motif_co = np.argsort(distance_profile)[-1]
        motif_se = np.argsort(distance_profile)[-2]
        motif = dict (
            distance_profile = distance_profile,
            motif_position = motif_co*m,
            motif_sec = motif_se*m,
            average_profile = average_profile
        )
        return motif

    def __init__(self, data_cleanned):
         self.data_cleanned = data_cleanned

    def motifs_discovery_v2(self, user_type, motifs_range):
        motif_data =  self.data_cleanned[user_type][motifs_range[0]:] if motifs_range[1] ==0 else self.data_cleanned[user_type][motifs_range[0]:motifs_range[1]]
        targeted_series = motif_data[user_type] if user_type.find('Solar') != -1 else motif_data['consumption']
        ## No special attention applied
        motifs = self.motif_dtw_24_w_dy(targeted_series, 96, np.ones((96)))
        motif_pattern = targeted_series[motifs['motif_position']:motifs['motif_position']+96]
        print('Refined Motif at: ',motif_data.index[motifs['motif_position']])
        temperature_motif = motif_data['temperature (degC)'][motifs['motif_position']:motifs['motif_position']+96]
        cloudcover_motif = motif_data ['total_cloud_cover (0-1)'][motifs['motif_position']:motifs['motif_position']+96]
        humidity_motif = motif_data ['relative_humidity ((0-1))'][motifs['motif_position']:motifs['motif_position']+96]
        radiation_motif = motif_data ['surface_solar_radiation (W/m^2)'][motifs['motif_position']:motifs['motif_position']+96]
        sealevel_motif = motif_data ['mean_sea_level_pressure (Pa)'][motifs['motif_position']:motifs['motif_position']+96]
        wind_motif = motif_data ['wind_speed (m/s)'][motifs['motif_position']:motifs['motif_position']+96]
        times_motif = pd.CategoricalIndex(motif_data.index.time).codes.astype(float) 
        return motifs, motif_data, motif_pattern, temperature_motif, cloudcover_motif, humidity_motif, radiation_motif, sealevel_motif, wind_motif, times_motif

    def motifs_discovery(self, user_type, motifs_range):
        motif_data =  self.data_cleanned[user_type][motifs_range[0]:] if motifs_range[1] ==0 else self.data_cleanned[user_type][motifs_range[0]:motifs_range[1]]
        targeted_series = motif_data[user_type] if user_type.find('Solar') != -1 else motif_data['consumption']
        ## No special attention applied
        motifs = self.motif_dtw_24_w_dy(targeted_series, 96, np.ones((96)))
        motif_pattern = targeted_series[motifs['motif_position']:motifs['motif_position']+96]
        print('Refined Motif at: ',motif_data.index[motifs['motif_position']])
        temperature_motif = motif_data['temperature (degC)'][motifs['motif_position']:motifs['motif_position']+96]
        cloudcover_motif = motif_data ['total_cloud_cover (0-1)'][motifs['motif_position']:motifs['motif_position']+96]
        humidity_motif = motif_data ['relative_humidity ((0-1))'][motifs['motif_position']:motifs['motif_position']+96]
        radiation_motif = motif_data ['surface_solar_radiation (W/m^2)'][motifs['motif_position']:motifs['motif_position']+96]
        times_motif = pd.CategoricalIndex(motif_data.index.time).codes.astype(float) 
        return motifs, motif_data, motif_pattern, temperature_motif, cloudcover_motif, humidity_motif, radiation_motif, times_motif

class NN_model:
    '''NN based model for prediction (ResNet and 1-DCNN)'''
    ### Input is the training data and window size
    def __init__(self, x_train_cnn, y_train, m):
         self.input_shape_1 = x_train_cnn.shape[1]
         self.input_shape_2 = x_train_cnn.shape[2]
         self.output_shape = y_train.shape[1]
         self.m = m

    def identity_block(self, x, filter):
        # copy tensor to variable called x_skip
        x_skip = x
        # Layer 1
        x = Conv1D(filter,1, strides=1, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.Activation('relu')(x)
        # x = Dropout(0.3)(x)
        # Layer 2
        x = Conv1D(filter,3, strides=1, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        # Add Residue
        x = tf.keras.layers.Add()([x, x_skip])     
        x = tf.keras.layers.Activation('relu')(x)
        return x
    def convolutional_block(self, x, filter):
        # copy tensor to variable called x_skip
        x_skip = x
        # Layer 1
        x = Conv1D(filter,1, strides=1, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.Activation('relu')(x)
        # x = Dropout(0.3)(x)
        # Layer 2
        x = Conv1D(filter,3, strides=1, padding='same')(x)
        # Processing Residue with conv(1,1)
        x_skip = Conv1D(filter,1, strides=1, padding='same')(x_skip)
        x = BatchNormalization(momentum=0.8)(x)
        # Add Residue
        x = tf.keras.layers.Add()([x, x_skip])     
        x = tf.keras.layers.Activation('relu')(x)
        return x
    ### ResNet model (Projection layer is 1D-CNN)
    def ResNet(self, block_layers = [2, 2, 2, 2], error_matric= 'mse'):
        # Step 1 (Setup Input Layer)
        x_input = tf.keras.layers.Input((self.input_shape_1, self.input_shape_2))
        # Step 2 (Initial Conv layer along with maxPool)
        filter_size = self.m
        x = Conv1D(filter_size,1, strides=1, padding='same')(x_input)
        x = MaxPooling1D(pool_size=4)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = MaxPooling1D(pool_size=4)(x)
        x = Dropout(0.3)(x)
        # Define size of sub-blocks and initial filter size
        ### Other options:
        # block_layers = [3, 4, 23, 3]
        # Step 3 Add the Resnet Blocks
        for i in range(4):
            if i == 0:
                # For sub-block 1 Residual/Convolutional block not needed
                for j in range(block_layers[i]):
                    x = self.identity_block(x, filter_size)
            else:
                # One Residual/Convolutional Block followed by Identity blocks
                # The filter size will go on increasing by a factor of 2
                filter_size = filter_size*2
                x = self.convolutional_block(x, filter_size)
                for j in range(block_layers[i] - 1):
                    x = self.identity_block(x, filter_size)
        # Step 4 End Dense Network
        # x = MaxPooling1D(pool_size=4)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation = 'relu')(x)
        x = Dropout(0.3)(x)
        x = tf.keras.layers.Dense(self.output_shape, activation='relu')(x)
        model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet")
        model.compile(optimizer='adam', loss=error_matric)
        return model

    ### Transformer
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res
    def Transformer(
        self,
        head_size = 256,
        num_heads = 4,
        ff_dim = 4,
        num_transformer_blocks = 4,
        mlp_units = [128],
        dropout=0.2,
        mlp_dropout=0.25,
        error_matric = 'mse'
    ):
        input_shape = (self.input_shape_1, self.input_shape_2)
        output_shape = self.output_shape
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(output_shape, activation='relu')(x)
        model = keras.Model(inputs, outputs, name = 'Transformer')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=error_matric)
        return model
    ### Time series CNN model
    def CNN_series(self, error_matric= 'mse'):
        model = Sequential()
        model.add(Conv1D(filters=48, kernel_size=3, activation='relu', input_shape=(self.input_shape_1, self.input_shape_2)))
        # model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(self.output_shape, activation='linear')) 
        model.compile(optimizer='adam', loss=error_matric)
        return model


class building_methods:
    '''Building Prediction class'''
    def __init__(self, dict_df):
         self.dict_df = dict_df

    def building_decomposition(self, training_type):
        ### STL decomposition on picked buildings
        
        for key in self.dict_df:
            
            if key.find(training_type) == -1:
                continue
                
            stl = STL(self.dict_df[key][key][-4*24*31*3:], period=96*7, seasonal=96+1, robust=False)
            result = stl.fit()
            seasonal, trend, resid = result.seasonal, result.trend, result.resid
            
            self.dict_df[key]['res_' + key] = self.dict_df[key][key].copy()
            self.dict_df[key]['res_' + key][-4*24*31*3:] = resid.values
            
            self.dict_df[key]['tre_' + key] = self.dict_df[key][key].copy()
            self.dict_df[key]['tre_' + key][-4*24*31*3:] = trend.values
            
            self.dict_df[key]['sea_' + key] = self.dict_df[key][key].copy()
            self.dict_df[key]['sea_' + key][-4*24*31*3:] = seasonal.values
            
        return self.dict_df

    def split_training_dataframe(self, training_type='Solar0', training_length=24*4*31):
        
        df = self.dict_df[training_type].copy()
        df['YEAR'] = df.index.year
        df['MONTH'] = df.index.month
        df['WEEK'] = df.index.isocalendar().week
        df['DATE'] = df.index.day
        df['HOUR'] = df.index.hour
        df['TIME'] = df.index.time
        
        df['TIME_CAT'] = df['TIME'].astype('category').cat.codes.astype(float)
    #     df['TIME_GROUP'] = np.maximum(np.sin(2*np.pi*(df['TIME_CAT'] + 8)/96), 0)
        
        df.loc[(df['TIME_CAT'] <= 36) | (df['TIME_CAT'] >= 84), 'TIME_GROUP'] = 3
        df.loc[(df['TIME_CAT'] > 36) & (df['TIME_CAT'] <= 42), 'TIME_GROUP'] = 2
        df.loc[(df['TIME_CAT'] > 42) & (df['TIME_CAT'] <= 76), 'TIME_GROUP'] = 1
        df.loc[(df['TIME_CAT'] > 76) & (df['TIME_CAT'] < 84), 'TIME_GROUP'] = 0
        
        df['WEEKDAY'] = df.index.weekday
        df['ISWEEKEND'] = np.where((df['WEEKDAY'] == 5) | (df['WEEKDAY'] == 6), True, False)
        df.dropna(inplace=True)
        
        df_training = df[-training_length:]
        
        return df_training, df

    def get_training_parameter(self, df_training, training_type, component='original'):
        
        # Training data
        if component == 'residual':
            y_training = np.array(df_training['res_' + training_type])
        elif component == 'trend':
            y_training = np.array(df_training['tre_' + training_type])
        elif component == 'seasonal':
            y_training = np.array(df_training['sea_' + training_type])
        elif component == 'original':
            y_training = np.array(df_training[training_type])

        dew_tempC_tr = df_training['dewpoint_temperature (degC)']
        tempC_tr = df_training['temperature (degC)']
        radiation_tr = df_training['surface_solar_radiation (W/m^2)']
        thermal_radiation_tr = df_training['surface_thermal_radiation (W/m^2)']
        cloudcover_tr = df_training['total_cloud_cover (0-1)']
        humidity_tr = df_training['relative_humidity ((0-1))']
        hour_tr = df_training['HOUR'].astype('category').cat.codes.astype(float)
        weekday_tr = df_training['WEEKDAY'].astype('category').cat.codes.astype(float)
        isweekend_tr = df_training['ISWEEKEND'].astype('category').cat.codes.astype(float)
        time_tr = df_training['TIME'].astype('category').cat.codes.astype(float)
        time_group_tr = df_training['TIME_GROUP']
        week_tr = df_training['WEEK']
        
        training_parameter_RF = {"dew_tempC": dew_tempC_tr,
                                "tempC": tempC_tr,
                                "radiation": radiation_tr,
                                "thermal_radiation": thermal_radiation_tr,
                                "cloudcover": cloudcover_tr,
                                "humidity": humidity_tr,
                                "hour": hour_tr,
                                "weekday": weekday_tr,
                                "isweekend": isweekend_tr,
                                "time": time_tr,
                                "time_group": time_group_tr,
                                "week": week_tr}

        training_parameter_SVM = {"dew_tempC": dew_tempC_tr,
                                "dew_tempC_2": dew_tempC_tr**2,
                                "tempC": tempC_tr,
                                "tempC_2": tempC_tr**2,
                                "radiation": radiation_tr,
                                "thermal_radiation": thermal_radiation_tr,
                                "cloudcover": cloudcover_tr,
                                "humidity": humidity_tr,
                                "hour": hour_tr,
                                "weekday": weekday_tr,
                                "weekday_2": weekday_tr**2,
                                "weekday_3": weekday_tr**3,
                                "weekday_4": weekday_tr**4,
                                "weekday_5": weekday_tr**5,
                                "isweekend": isweekend_tr,
                                "time": time_tr,
                                "time_group": time_group_tr,
                                "time_group_2": time_group_tr**2,
                                "week": week_tr}
        
        return training_parameter_RF, training_parameter_SVM, y_training

    def get_prediction_parameter(self, df_prediction, training_type, component='original'):
        
        # Predicting data
        
        dew_tempC_pr = df_prediction['dewpoint_temperature (degC)']
        tempC_pr = df_prediction['temperature (degC)']
        radiation_pr = df_prediction['surface_solar_radiation (W/m^2)']
        thermal_radiation_pr = df_prediction['surface_thermal_radiation (W/m^2)']
        cloudcover_pr = df_prediction['total_cloud_cover (0-1)']
        humidity_pr = df_prediction['relative_humidity ((0-1))']
        hour_pr = df_prediction['HOUR'].astype('category').cat.codes.astype(float)
        weekday_pr = df_prediction['WEEKDAY'].astype('category').cat.codes.astype(float)
        isweekend_pr = df_prediction['ISWEEKEND'].astype('category').cat.codes.astype(float)
        time_pr = df_prediction['TIME'].astype('category').cat.codes.astype(float)
        time_group_pr = df_prediction['TIME_GROUP']
        week_pr = df_prediction['WEEK']
        
        prediction_parameter_RF = {"dew_tempC": dew_tempC_pr,
                                    "tempC": tempC_pr,
                                    "radiation": radiation_pr,
                                    "thermal_radiation": thermal_radiation_pr,
                                    "cloudcover": cloudcover_pr,
                                    "humidity": humidity_pr,
                                    "hour": hour_pr,
                                    "weekday": weekday_pr,
                                    "isweekend": isweekend_pr,
                                    "time": time_pr,
                                    "time_group": time_group_pr,
                                    "week": week_pr}

        prediction_parameter_SVM = {"dew_tempC": dew_tempC_pr,
                                    "dew_tempC_2": dew_tempC_pr**2,
                                    "tempC": tempC_pr,
                                    "tempC_2": tempC_pr**2,
                                    "radiation": radiation_pr,
                                    "thermal_radiation": thermal_radiation_pr,
                                    "cloudcover": cloudcover_pr,
                                    "humidity": humidity_pr,
                                    "hour": hour_pr,
                                    "weekday": weekday_pr,
                                    "weekday_2": weekday_pr**2,
                                    "weekday_3": weekday_pr**3,
                                    "weekday_4": weekday_pr**4,
                                    "weekday_5": weekday_pr**5,
                                    "isweekend": isweekend_pr,
                                    "time": time_pr,
                                    "time_group": time_group_pr,
                                    "time_group_2": time_group_pr**2,
                                    "week": week_pr}
        
        return prediction_parameter_RF, prediction_parameter_SVM

    def SVM_training(self, x_training_SVM, y_training):
        
        gamma_train = 0.01
        C_train = 0.1
        regression = SVR(kernel='rbf', gamma=gamma_train, C=C_train)
        regression = make_pipeline(StandardScaler(), regression)
        regression.fit(x_training_SVM, y_training)
        return regression

    def RF_training(self, x_training_RF, y_training):
        rf = RandomForestRegressor(n_estimators=1000,
                                min_samples_split=8, random_state=42,
                                n_jobs=-1)  # Use maximum number of cores.
        # rf = make_pipeline(StandardScaler(), rf)
        rf.fit(x_training_RF, y_training)
        return rf

    def GB_training(self, x_training_RF, y_training):
        rf = GradientBoostingRegressor(n_estimators=1000,
                                min_samples_split=8, random_state=42, learning_rate=0.1, loss='lad')  # Use maximum number of cores.
        # rf = make_pipeline(StandardScaler(), rf)
        rf.fit(x_training_RF, y_training)
        return rf

    def prepare_unseen_dataframe(self, training_type='Building0'):
        
        df = self.dict_df['weather']
        start_date = pd.to_datetime('2020-11-01')
        end_date = start_date + timedelta(days=30)
        mask = (df.index >= start_date) & (df.index < end_date)
        df = df[mask]
        
        time_zone_change = pd.to_datetime('2020-10-03 16:00')
        df = df.reset_index()
        pd.options.mode.chained_assignment = None
        for i in range(len(df)):
            if df['datetime (UTC)'][i] >= time_zone_change:
                df['datetime (UTC)'][i] = df['datetime (UTC)'][i] + timedelta(hours=1)
        df = df.set_index('datetime (UTC)')
        
        df['YEAR'] = df.index.year
        df['MONTH'] = df.index.month
        df['WEEK'] = df.index.isocalendar().week
        df['DATE'] = df.index.day
        df['HOUR'] = df.index.hour
        df['TIME'] = df.index.time
        
        df['TIME_CAT'] = df['TIME'].astype('category').cat.codes.astype(float)
        df.loc[(df['TIME_CAT'] <= 36) | (df['TIME_CAT'] >= 84), 'TIME_GROUP'] = 3
        df.loc[(df['TIME_CAT'] > 36) & (df['TIME_CAT'] <= 42), 'TIME_GROUP'] = 2
        df.loc[(df['TIME_CAT'] > 42) & (df['TIME_CAT'] <= 76), 'TIME_GROUP'] = 1
        df.loc[(df['TIME_CAT'] > 76) & (df['TIME_CAT'] < 84), 'TIME_GROUP'] = 0
        
        df['WEEKDAY'] = df.index.weekday
        df['ISWEEKEND'] = np.where((df['WEEKDAY'] == 5) | (df['WEEKDAY'] == 6), True, False)
        df.dropna(inplace=True)

        return df

    def holiday_halving(self, predicted_values, base_consumption):
        ### half the predicted values for holidays
        ### Holiday: 03 Nov 2020
        
        holiday_profile = {"Building1": 12,
                        "Building3": 350,
                        "Building6": 29 }

        for i in range(96*2-15, 96*2+35):
            if predicted_values[i] >= base_consumption:
                predicted_values[i] = base_consumption + (predicted_values[i] - base_consumption)/2
        
        return predicted_values

    def train_unseen_STL(self, training_type, RF_input, SVM_input, training_length):

        component = ['residual', 'trend', 'seasonal', 'original']
        y_dict_SVM = dict()
        y_dict_RF = dict()
        y_dict_GB = dict()
        
        df_training, df = self.split_training_dataframe(training_type, training_length)
        
        df_unseen = self.prepare_unseen_dataframe(training_type)
        
        for c in component:
            
            training_parameter_RF, training_parameter_SVM, y_training = self.get_training_parameter(df_training, training_type, c)

            x_training_SVM = np.column_stack(([training_parameter_SVM[x] for x in SVM_input]))
            x_training_RF = np.column_stack(([training_parameter_RF[x] for x in RF_input]))

            SVM_regression = self.SVM_training(x_training_SVM, y_training)
            RF_regression = self.RF_training(x_training_RF, y_training)
            GB_regression = self.GB_training(x_training_RF, y_training)

            prediction_parameter_RF, prediction_parameter_SVM = self.get_prediction_parameter(df_unseen, training_type, c)
            x_prediction_SVM = np.column_stack(([prediction_parameter_SVM[x] for x in SVM_input]))
            x_prediction_RF = np.column_stack(([prediction_parameter_RF[x] for x in RF_input]))

            # Perform 3 different techniques on all components
            y_dict_SVM[c] = SVM_regression.predict(x_prediction_SVM)
            y_dict_RF[c] = RF_regression.predict(x_prediction_RF)
            y_dict_GB[c] = GB_regression.predict(x_prediction_RF)
            
    #         if c == 'trend':
    #             scaler = StandardScaler()
    #             scaler.fit(x_training_RF)
    #             x_training_RF_scaled = scaler.transform(x_training_RF)
    #             x_prediction_RF_scaled = scaler.transform(x_prediction_RF)
    #             model = ARIMA(y_training[-24*4*31:], order=(25,1,0), exog=x_training_RF_scaled[-24*4*31:])
    #             model_fit = model.fit()
    #             output = model_fit.forecast(len(df_unseen), exog=x_prediction_RF_scaled)

    #     y_predict = y_dict_SVM['residual'] + y_dict_RF['trend'] + dict_df[training_type]['sea_' + training_type][-24*4*31:].values
        
        return y_dict_RF, y_dict_SVM, y_dict_GB

    def train_unseen_tree(self, training_type, RF_input, SVM_input, training_length):

        df_training, df = self.split_training_dataframe(training_type, training_length)
        training_parameter_RF, training_parameter_SVM, y_training = self.get_training_parameter(df_training, training_type)

        x_training_SVM = np.column_stack(([training_parameter_SVM[x] for x in SVM_input]))
        x_training_RF = np.column_stack(([training_parameter_RF[x] for x in RF_input]))

        SVM_regression = self.SVM_training(x_training_SVM, y_training)
        RF_regression = self.RF_training(x_training_RF, y_training)
        GB_regression = self.GB_training(x_training_RF, y_training)
        
        df_unseen = self.prepare_unseen_dataframe(training_type)
        prediction_parameter_RF, prediction_parameter_SVM = self.get_prediction_parameter(df_unseen, training_type)
        x_prediction_SVM = np.column_stack(([prediction_parameter_SVM[x] for x in SVM_input]))
        x_prediction_RF = np.column_stack(([prediction_parameter_RF[x] for x in RF_input]))
        
        y_predict_SVM = SVM_regression.predict(x_prediction_SVM)
        y_predict_RF = RF_regression.predict(x_prediction_RF)
        y_predict_GB = GB_regression.predict(x_prediction_RF)

        return y_predict_RF, y_predict_SVM, y_predict_GB

### Other functions
def find_index_v2(data_cleanned_v2, type, datetime):
    for i in range(len(data_cleanned_v2[type])):
        if data_cleanned_v2[type].index[i] == datetime:
            return i

def check_loads_plot(data_cleanned_v2, check_name):
    time_series = data_cleanned_v2[check_name].index
    real_value = data_cleanned_v2[check_name]['consumption'] if check_name.find('Building') != -1 else data_cleanned_v2[check_name][check_name]
    fig = go.Figure()
    fig.add_trace(
            go.Scatter(
                x= time_series, y=real_value, line_width = 0.8, opacity = 0.7, name = check_name
            )
        )
    fig.layout.title = check_name+" Loads Data"
    fig.layout.xaxis.title = "Time"
    fig.layout.yaxis.title = "ELectricity (kW)"
    fig.layout.plot_bgcolor='rgba(0,0,0,0)'
    fig.layout.font.family="Times New Roman" ## "Times New Roman Black"
    fig.layout.font.size = 20
    fig.layout.width = 1000
    fig.layout.height = 500
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True,showgrid=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showgrid=False)
    fig.show()