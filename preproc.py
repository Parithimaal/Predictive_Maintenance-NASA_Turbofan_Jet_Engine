import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    """
    Contains the functions for preprocessing the NASA Turbofan dataframes
    """
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

    @staticmethod
    def normalize(df, cols):
        df_norm = df.copy()
        df_norm[cols] = minmaxscaler.fit_transform(df_norm[(cols)])
        return df_norm