"""
Feature Engineering Module
Advanced feature creation and transformation techniques
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureEngineer:
    """
    Feature Engineering class for creating and transforming features
    """
    
    def __init__(self, dataframe):
        """
        Initialize with a pandas DataFrame
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            Input dataframe
        """
        self.df = dataframe.copy()
        self.df_engineered = None
        self.scaler = None
        self.feature_importance = {}
        
    def create_interaction_features(self, feature_cols):
        """
        Create interaction features (multiplicative combinations)
        
        Parameters:
        -----------
        feature_cols : list
            List of columns to create interactions from
        """
        print("\n" + "="*80)
        print("CREATING INTERACTION FEATURES")
        print("="*80)
        
        self.df_engineered = self.df.copy()
        created_features = []
        
        # Two-way interactions
        for i in range(len(feature_cols)):
            for j in range(i+1, len(feature_cols)):
                feature_name = f"{feature_cols[i]}_x_{feature_cols[j]}"
                self.df_engineered[feature_name] = (
                    self.df_engineered[feature_cols[i]] * 
                    self.df_engineered[feature_cols[j]]
                )
                created_features.append(feature_name)
                print(f"✓ Created: {feature_name}")
        
        print(f"\nTotal interaction features created: {len(created_features)}")
        return created_features
    
    def create_polynomial_features(self, feature_cols, degree=2):
        """
        Create polynomial features
        
        Parameters:
        -----------
        feature_cols : list
            Columns to create polynomial features from
        degree : int
            Degree of polynomial features
        """
        print("\n" + "="*80)
        print(f"CREATING POLYNOMIAL FEATURES (Degree {degree})")
        print("="*80)
        
        if self.df_engineered is None:
            self.df_engineered = self.df.copy()
        
        created_features = []
        
        for col in feature_cols:
            for d in range(2, degree + 1):
                feature_name = f"{col}_pow_{d}"
                self.df_engineered[feature_name] = self.df_engineered[col] ** d
                created_features.append(feature_name)
                print(f"✓ Created: {feature_name}")
        
        print(f"\nTotal polynomial features created: {len(created_features)}")
        return created_features
    
    def create_ratio_features(self, feature_cols):
        """
        Create ratio features between different advertising channels
        
        Parameters:
        -----------
        feature_cols : list
            Columns to create ratios from
        """
        print("\n" + "="*80)
        print("CREATING RATIO FEATURES")
        print("="*80)
        
        if self.df_engineered is None:
            self.df_engineered = self.df.copy()
        
        created_features = []
        
        for i in range(len(feature_cols)):
            for j in range(len(feature_cols)):
                if i != j:
                    feature_name = f"{feature_cols[i]}_to_{feature_cols[j]}_ratio"
                    # Add small constant to avoid division by zero
                    self.df_engineered[feature_name] = (
                        self.df_engineered[feature_cols[i]] / 
                        (self.df_engineered[feature_cols[j]] + 1e-6)
                    )
                    created_features.append(feature_name)
                    print(f"✓ Created: {feature_name}")
        
        print(f"\nTotal ratio features created: {len(created_features)}")
        return created_features
    
    def create_aggregate_features(self, feature_cols):
        """
        Create aggregate features (sum, mean, etc.)
        
        Parameters:
        -----------
        feature_cols : list
            Columns to aggregate
        """
        print("\n" + "="*80)
        print("CREATING AGGREGATE FEATURES")
        print("="*80)
        
        if self.df_engineered is None:
            self.df_engineered = self.df.copy()
        
        created_features = []
        
        # Total advertising spend
        feature_name = "Total_Advertising_Spend"
        self.df_engineered[feature_name] = self.df_engineered[feature_cols].sum(axis=1)
        created_features.append(feature_name)
        print(f"✓ Created: {feature_name}")
        
        # Average advertising spend
        feature_name = "Average_Advertising_Spend"
        self.df_engineered[feature_name] = self.df_engineered[feature_cols].mean(axis=1)
        created_features.append(feature_name)
        print(f"✓ Created: {feature_name}")
        
        # Maximum channel spend
        feature_name = "Max_Channel_Spend"
        self.df_engineered[feature_name] = self.df_engineered[feature_cols].max(axis=1)
        created_features.append(feature_name)
        print(f"✓ Created: {feature_name}")
        
        # Minimum channel spend
        feature_name = "Min_Channel_Spend"
        self.df_engineered[feature_name] = self.df_engineered[feature_cols].min(axis=1)
        created_features.append(feature_name)
        print(f"✓ Created: {feature_name}")
        
        # Standard deviation of spend across channels
        feature_name = "Spend_Std"
        self.df_engineered[feature_name] = self.df_engineered[feature_cols].std(axis=1)
        created_features.append(feature_name)
        print(f"✓ Created: {feature_name}")
        
        print(f"\nTotal aggregate features created: {len(created_features)}")
        return created_features
    
    def scale_features(self, feature_cols, method='standard'):
        """
        Scale features using different scaling methods
        
        Parameters:
        -----------
        feature_cols : list
            Columns to scale
        method : str
            'standard', 'minmax', or 'robust'
        """
        print("\n" + "="*80)
        print(f"SCALING FEATURES ({method.upper()} scaling)")
        print("="*80)
        
        if self.df_engineered is None:
            self.df_engineered = self.df.copy()
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            print(f"✗ Unknown scaling method: {method}")
            return None
        
        # Create scaled feature names
        scaled_features = [f"{col}_scaled" for col in feature_cols]
        
        # Fit and transform
        self.df_engineered[scaled_features] = self.scaler.fit_transform(
            self.df_engineered[feature_cols]
        )
        
        print(f"✓ Scaled {len(feature_cols)} features using {method} method")
        
        # Display scaling statistics
        print("\nScaling Statistics:")
        print("-" * 80)
        for original, scaled in zip(feature_cols, scaled_features):
            print(f"{original}:")
            print(f"  Original - Mean: {self.df_engineered[original].mean():.2f}, "
                  f"Std: {self.df_engineered[original].std():.2f}")
            print(f"  Scaled   - Mean: {self.df_engineered[scaled].mean():.2f}, "
                  f"Std: {self.df_engineered[scaled].std():.2f}")
        
        return scaled_features
    
    def analyze_correlations(self, target_col, save_path=None):
        """
        Analyze correlations between features and target
        
        Parameters:
        -----------
        target_col : str
            Target column name
        save_path : str or Path, optional
            Path to save correlation heatmap
        """
        if self.df_engineered is None:
            df_to_analyze = self.df
        else:
            df_to_analyze = self.df_engineered
        
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)
        
        # Calculate correlation matrix
        corr_matrix = df_to_analyze.corr()
        
        # Get correlations with target
        target_corr = corr_matrix[target_col].sort_values(ascending=False)
        
        print(f"\nTop 15 Features Correlated with {target_col}:")
        print("-" * 80)
        print(target_corr.head(15))
        
        # Visualize correlation matrix
        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Correlation heatmap saved to {save_path}")
        
        plt.show()
        
        return target_corr
    
    def select_features(self, target_col, method='correlation', threshold=0.1, top_n=None):
        """
        Select most important features
        
        Parameters:
        -----------
        target_col : str
            Target column name
        method : str
            Feature selection method ('correlation', 'variance')
        threshold : float
            Correlation threshold for feature selection
        top_n : int, optional
            Select top N features
        """
        print("\n" + "="*80)
        print(f"FEATURE SELECTION ({method.upper()} method)")
        print("="*80)
        
        if self.df_engineered is None:
            df_to_select = self.df
        else:
            df_to_select = self.df_engineered
        
        if method == 'correlation':
            # Calculate correlations
            correlations = df_to_select.corr()[target_col].abs().sort_values(ascending=False)
            
            # Remove target itself
            correlations = correlations[correlations.index != target_col]
            
            if top_n:
                selected_features = list(correlations.head(top_n).index)
            else:
                selected_features = list(correlations[correlations > threshold].index)
            
            print(f"\nSelected {len(selected_features)} features with correlation > {threshold}:")
            print("-" * 80)
            for feature in selected_features:
                print(f"  {feature}: {correlations[feature]:.4f}")
        
        elif method == 'variance':
            # Calculate variance
            variances = df_to_select.var().sort_values(ascending=False)
            
            if top_n:
                selected_features = list(variances.head(top_n).index)
            else:
                selected_features = list(variances[variances > threshold].index)
            
            if target_col in selected_features:
                selected_features.remove(target_col)
            
            print(f"\nSelected {len(selected_features)} features:")
            print("-" * 80)
            for feature in selected_features:
                print(f"  {feature}: {variances[feature]:.4f}")
        
        return selected_features
    
    def get_engineered_data(self):
        """
        Return the engineered dataset
        """
        if self.df_engineered is None:
            return self.df
        return self.df_engineered