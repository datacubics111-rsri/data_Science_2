"""
Data Preprocessing Module
This module handles all data cleaning and preprocessing operations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    A comprehensive class for data preprocessing operations
    following IBM's data science best practices
    """
    
    def __init__(self, filepath):
        """
        Initialize the preprocessor with data file path
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the CSV file
        """
        self.filepath = filepath
        self.df = None
        self.df_processed = None
        self.preprocessing_report = {}
        
    def load_data(self):
        """
        Load data from CSV file with error handling
        """
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"✓ Data loaded successfully!")
            print(f"  Shape: {self.df.shape}")
            print(f"  Columns: {list(self.df.columns)}")
            return self.df
        except FileNotFoundError:
            print(f"✗ Error: File not found at {self.filepath}")
            return None
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return None
    
    def initial_exploration(self):
        """
        Perform initial data exploration
        
        Returns:
        --------
        dict : Dictionary containing exploration results
        """
        if self.df is None:
            print("✗ Please load data first using load_data()")
            return None
        
        print("\n" + "="*80)
        print("INITIAL DATA EXPLORATION")
        print("="*80)
        
        # Basic Information
        print("\n1. DATASET OVERVIEW")
        print("-" * 80)
        print(f"Number of Records: {len(self.df)}")
        print(f"Number of Features: {len(self.df.columns)}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Display first few rows
        print("\n2. FIRST 5 RECORDS")
        print("-" * 80)
        print(self.df.head())
        
        # Data types
        print("\n3. DATA TYPES")
        print("-" * 80)
        print(self.df.dtypes)
        
        # Statistical Summary
        print("\n4. STATISTICAL SUMMARY")
        print("-" * 80)
        print(self.df.describe())
        
        # Missing values
        print("\n5. MISSING VALUES ANALYSIS")
        print("-" * 80)
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Percentage': missing_percent
        })
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        if missing_df['Missing_Count'].sum() == 0:
            print("✓ No missing values found!")
        
        # Duplicate records
        print("\n6. DUPLICATE RECORDS")
        print("-" * 80)
        duplicates = self.df.duplicated().sum()
        print(f"Number of duplicate records: {duplicates}")
        
        exploration_report = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': missing_df.to_dict(),
            'duplicates': duplicates,
            'summary_stats': self.df.describe().to_dict()
        }
        
        return exploration_report
    
    def detect_outliers(self, columns=None, method='iqr', threshold=1.5):
        """
        Detect outliers using IQR or Z-score method
        
        Parameters:
        -----------
        columns : list, optional
            Columns to check for outliers
        method : str
            'iqr' for Interquartile Range or 'zscore' for Z-score method
        threshold : float
            Threshold for outlier detection (1.5 for IQR, 3 for Z-score)
        """
        if self.df is None:
            print("✗ Please load data first")
            return None
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        print("\n" + "="*80)
        print("OUTLIER DETECTION")
        print("="*80)
        
        outliers_info = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col]))
                outliers = self.df[z_scores > threshold]
            
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.df)) * 100,
                'method': method
            }
            
            print(f"\n{col}:")
            print(f"  Outliers detected: {len(outliers)} ({outliers_info[col]['percentage']:.2f}%)")
            if method == 'iqr':
                print(f"  Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return outliers_info
    
    def visualize_distributions(self, save_path=None):
        """
        Visualize distribution of all numerical features
        
        Parameters:
        -----------
        save_path : str or Path, optional
            Path to save the figure
        """
        if self.df is None:
            print("✗ Please load data first")
            return
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        n_cols = len(numerical_cols)
        
        fig, axes = plt.subplots(n_cols, 2, figsize=(15, 5*n_cols))
        fig.suptitle('Distribution Analysis of Numerical Features', 
                     fontsize=16, fontweight='bold', y=1.002)
        
        for idx, col in enumerate(numerical_cols):
            # Histogram
            axes[idx, 0].hist(self.df[col], bins=30, edgecolor='black', alpha=0.7)
            axes[idx, 0].set_title(f'{col} - Histogram')
            axes[idx, 0].set_xlabel(col)
            axes[idx, 0].set_ylabel('Frequency')
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Box plot
            axes[idx, 1].boxplot(self.df[col])
            axes[idx, 1].set_title(f'{col} - Box Plot')
            axes[idx, 1].set_ylabel(col)
            axes[idx, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Distribution plots saved to {save_path}")
        
        plt.show()
    
    def clean_data(self, remove_duplicates=True, handle_missing='drop', 
                   handle_outliers='keep', outlier_method='iqr'):
        """
        Comprehensive data cleaning
        
        Parameters:
        -----------
        remove_duplicates : bool
            Whether to remove duplicate records
        handle_missing : str
            'drop', 'mean', 'median', or 'mode'
        handle_outliers : str
            'keep', 'remove', or 'cap'
        outlier_method : str
            'iqr' or 'zscore'
        """
        if self.df is None:
            print("✗ Please load data first")
            return None
        
        print("\n" + "="*80)
        print("DATA CLEANING PROCESS")
        print("="*80)
        
        self.df_processed = self.df.copy()
        initial_shape = self.df_processed.shape
        
        # 1. Remove duplicates
        if remove_duplicates:
            before = len(self.df_processed)
            self.df_processed.drop_duplicates(inplace=True)
            after = len(self.df_processed)
            removed = before - after
            print(f"\n1. Duplicate Removal: {removed} records removed")
        
        # 2. Handle missing values
        missing_count = self.df_processed.isnull().sum().sum()
        if missing_count > 0:
            print(f"\n2. Missing Values: {missing_count} found")
            
            if handle_missing == 'drop':
                self.df_processed.dropna(inplace=True)
                print(f"   Action: Dropped rows with missing values")
            
            elif handle_missing == 'mean':
                for col in self.df_processed.select_dtypes(include=[np.number]).columns:
                    self.df_processed[col].fillna(self.df_processed[col].mean(), inplace=True)
                print(f"   Action: Filled with mean values")
            
            elif handle_missing == 'median':
                for col in self.df_processed.select_dtypes(include=[np.number]).columns:
                    self.df_processed[col].fillna(self.df_processed[col].median(), inplace=True)
                print(f"   Action: Filled with median values")
        else:
            print(f"\n2. Missing Values: None found ✓")
        
        # 3. Handle outliers
        if handle_outliers != 'keep':
            print(f"\n3. Outlier Handling ({outlier_method} method):")
            numerical_cols = self.df_processed.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                if outlier_method == 'iqr':
                    Q1 = self.df_processed[col].quantile(0.25)
                    Q3 = self.df_processed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    if handle_outliers == 'remove':
                        before = len(self.df_processed)
                        self.df_processed = self.df_processed[
                            (self.df_processed[col] >= lower_bound) & 
                            (self.df_processed[col] <= upper_bound)
                        ]
                        after = len(self.df_processed)
                        print(f"   {col}: {before - after} outliers removed")
                    
                    elif handle_outliers == 'cap':
                        self.df_processed[col] = self.df_processed[col].clip(
                            lower=lower_bound, upper=upper_bound
                        )
                        print(f"   {col}: Outliers capped to [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        final_shape = self.df_processed.shape
        
        print(f"\n" + "-"*80)
        print(f"CLEANING SUMMARY:")
        print(f"  Initial shape: {initial_shape}")
        print(f"  Final shape: {final_shape}")
        print(f"  Records removed: {initial_shape[0] - final_shape[0]}")
        print(f"  Data retained: {(final_shape[0]/initial_shape[0])*100:.2f}%")
        print("="*80)
        
        return self.df_processed
    
    def save_processed_data(self, filepath):
        """
        Save processed data to CSV
        
        Parameters:
        -----------
        filepath : str or Path
            Path to save the processed data
        """
        if self.df_processed is None:
            print("✗ No processed data to save. Run clean_data() first")
            return False
        
        try:
            self.df_processed.to_csv(filepath, index=False)
            print(f"✓ Processed data saved to {filepath}")
            return True
        except Exception as e:
            print(f"✗ Error saving data: {str(e)}")
            return False