import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    """
    Contains the functions for preprocessing the NASA Turbofan dataframes
    """
    def __init__(self, saved_min_max=MinMaxScaler(), saved_std=StandardScaler()):
        self.minmaxscaler = saved_min_max
        self.stdscaler = saved_std
        self.sensor_cols = []
        self.op_setting_cols = []
        self.other_cols = []

    def rename_columns(self, df):# Change in main
        self.sensor_cols = [f"s{num}" for num in range(1, 24)]
        self.op_setting_cols = [f"ops{num}" for num in range(1, 4)]
        self.other_cols = ["unit", "cycles"]

        column_names = self.other_cols + self.op_setting_cols + self.sensor_cols
        df.columns = column_names
        return df
    
    def get_sensor_cols(self):
        return self.sensor_cols
    
    def get_op_setting_cols(self):
        return self.op_setting_cols
    
    def get_other_cols(self):
        return self.other_cols
    
    @staticmethod
    def dropna(df):
        df = df.dropna(axis=1, how='all')
        return df

    
    @staticmethod
    def drop_no_info_cols(df):
        no_info_cols = []
        for col in df.describe().columns:
            if df.describe().loc['min', col] == df.describe().loc['max', col]:
                no_info_cols.append(col)
        return df.drop(columns=no_info_cols)
    
    def create_rul_col(self, df):# Change in main
        rul_col = df.groupby('unit')['cycles'].transform('max') - df['cycles']
        df.insert(loc=2, column='rul', value=rul_col)
        return df

    def drop_unit(self,df):
        df = df.drop(columns=['unit'])
        return df

    def update_col_groups(self, df):
        self.sensor_cols = [col for col in df.columns if col.startswith('s')]
        self.op_setting_cols = [col for col in df.columns if col.startswith('ops')]
        self.other_cols = [col for col in df.columns if col not in self.sensor_cols+self.op_setting_cols]
        return df
    
    def normalize_predictors(self, df, cols):
        df_norm = df.copy()
        df_norm[cols] = self.minmaxscaler.fit_transform(df_norm[(cols)])
        return df_norm

    def standardize_predictors(self, df, cols):
        df_std = df.copy()
        df_std[cols] = self.stdscaler.fit_transform(df_std[(cols)])
        return df_std
    
    def get_minmaxscaler(self):
        return self.minmaxscaler
    
    def get_stdscaler(self):
        return self.stdscaler
    
    @staticmethod
    def group_by_rul(df, cols):
        agg_dict = {
            item: 'mean'
            for item in cols
        }
        agg_dict['unit'] = 'count'
        agg_dict['cycles'] = 'mean'
        return df.groupby('rul').agg(agg_dict)
    
    @staticmethod
    def group_by_unit(df, cols):
        agg_dict = {
            item: 'mean'
            for item in cols
        }
        agg_dict['cycles'] = 'mean'
        return df.groupby('unit').agg(agg_dict)
    
class Charting:
    
    @staticmethod
    def plot_avg_sensor_data(grouped_train_norm, sensor_cols):
        y_columns = sensor_cols
        fig = px.line(grouped_train_norm, x=grouped_train_norm.index, y=y_columns, title="Average Sensor values over all units vs RUL")
        return 
        

    @staticmethod
    def plot_corr_heatmap(grouped_df):
        plt.figure(figsize=(12, 10))
        sns.heatmap(np.round(grouped_df.reset_index().corr(), 2), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        return plt
    
    @staticmethod
    def plot_max_cycles_dist(train_df):
        max_cycles = train_df.groupby('unit').max('cycles').sort_values(by='cycles', ascending=True).reset_index().cycles
        percentiles = [25, 50, 75]  # 25th, 50th (median), and 75th percentiles
        percentile_values = np.percentile(max_cycles, percentiles)
        fig = px.histogram(max_cycles)
        colors = ['red', 'green', 'purple']
        line_names = ['p25', 'p50', 'p75']
        for i, percentile in enumerate(percentile_values):
            fig.add_vline(
                x=percentile,
                line_color=colors[i],
                line_dash='dash',
                line_width=2,
                annotation_text=f"{line_names[i]}",
                annotation_position="top"
            )
        fig.update_layout(title_text='Histogram of Max cycle of each unit')
        return fig
    
