import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    """
    Contains the functions for preprocessing the NASA Turbofan dataframes
    """
    def __init__(self, saved_scaler=MinMaxScaler()):
        self.minmaxscaler = saved_scaler

    @staticmethod
    def rename_columns(df):
        """
        Renames the columns of the dataframe
        """
        sensor_cols = [f"s{num}" for num in range(1, 24)]
        op_setting_cols = [f"ops{num}" for num in range(1, 4)]
        other_cols = ["unit", "cycles"]
        column_names = other_cols + op_setting_cols + sensor_cols
        df.columns = column_names
        return df
    
    @staticmethod
    def drop_no_info_cols(df):
        no_info_cols = []
        for col in df.describe().columns:
            if df.describe().loc['min', col] == df.describe().loc['max', col]:
                no_info_cols.append(col)
        return df.drop(columns=no_info_cols)
    
    @staticmethod
    def create_rul_col(df):
        rul_col = df.groupby('unit')['cycles'].transform('max') - df['cycles']
        df.insert(loc=2, column='rul', value=rul_col)
        return df

    @staticmethod
    def update_col_groups(df):
        sensor_cols = [col for col in df.columns if col.startswith('s')]
        op_setting_cols = [col for col in df.columns if col.startswith('ops')]
        other_cols = ["unit", "cycles", "rul"]
        return sensor_cols, op_setting_cols, other_cols

    def normalize(self, df, cols):
        df_norm = df.copy()
        df_norm[cols] = self.minmaxscaler.fit_transform(df_norm[(cols)])
        return df_norm
    
    @staticmethod
    def group_by_rul(df, cols):
        agg_dict = {
            item: 'mean'
            for item in cols
        }
        agg_dict['unit'] = 'count'
        agg_dict['cycles'] = 'mean'
        return df.groupby('rul').agg(agg_dict)
    
class Charting:
    
    @staticmethod
    def plot_avg_sensor_data(grouped_train_norm, sensor_cols):
        y_columns = sensor_cols
        return px.line(grouped_train_norm, x=grouped_train_norm.index, y=y_columns)
        

    @staticmethod
    def plot_corr_heatmap(grouped_df):
        plt.figure(figsize=(12, 10))
        sns.heatmap(np.round(grouped_df.reset_index().corr(), 2), annot=True, cmap='coolwarm')
        plt.title('Seaborn Correlation Matrix')
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
        return fig
    
