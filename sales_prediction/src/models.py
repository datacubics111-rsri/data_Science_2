"""
Machine Learning Models Module
Comprehensive modeling with multiple algorithms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class SalesPredictionModel:
    """
    Comprehensive Sales Prediction Model Class
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the model class
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, dataframe, target_col, feature_cols, test_size=0.2):
        """
        Prepare data for modeling
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            Input dataframe
        target_col : str
            Target column name
        feature_cols : list
            List of feature column names
        test_size : float
            Test set size
        """
        print("\n" + "="*80)
        print("DATA PREPARATION FOR MODELING")
        print("="*80)
        
        # Separate features and target
        X = dataframe[feature_cols]
        y = dataframe[target_col]
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"\nFeatures used: {len(feature_cols)}")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i}. {col}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"\nData Split:")
        print(f"  Training set: {self.X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
        print(f"  Test set: {self.X_test.shape[0]} samples ({test_size*100:.0f}%)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def initialize_models(self):
        """
        Initialize all regression models
        """
        print("\n" + "="*80)
        print("INITIALIZING MODELS")
        print("="*80)
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=self.random_state),
            'Lasso Regression': Lasso(random_state=self.random_state),
            'ElasticNet': ElasticNet(random_state=self.random_state),
            'Decision Tree': DecisionTreeRegressor(random_state=self.random_state),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingRegressor(random_state=self.random_state),
            'Support Vector Regression': SVR()
        }
        
        print(f"\n✓ {len(self.models)} models initialized:")
        for i, model_name in enumerate(self.models.keys(), 1):
            print(f"  {i}. {model_name}")
        
        return self.models
    
    def train_models(self):
        """
        Train all models and evaluate performance
        """
        print("\n" + "="*80)
        print("TRAINING MODELS")
        print("="*80)
        
        if not self.models:
            self.initialize_models()
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(self.y_train, y_train_pred)
            test_metrics = self._calculate_metrics(self.y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                       cv=5, scoring='r2')
            
            # Store results
            self.results[name] = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred
            }
            
            print(f"  ✓ Training complete")
            print(f"    Train R²: {train_metrics['r2']:.4f}")
            print(f"    Test R²: {test_metrics['r2']:.4f}")
            print(f"    CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        print("\n" + "="*80)
        print("✓ ALL MODELS TRAINED SUCCESSFULLY")
        print("="*80)
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate regression metrics
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        
        Returns:
        --------
        dict : Dictionary of metrics
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
    
    def compare_models(self):
        """
        Compare all models and display results
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Train_R2': [self.results[m]['train_metrics']['r2'] for m in self.results],
            'Test_R2': [self.results[m]['test_metrics']['r2'] for m in self.results],
            'Train_RMSE': [self.results[m]['train_metrics']['rmse'] for m in self.results],
            'Test_RMSE': [self.results[m]['test_metrics']['rmse'] for m in self.results],
            'Test_MAE': [self.results[m]['test_metrics']['mae'] for m in self.results],
            'Test_MAPE': [self.results[m]['test_metrics']['mape'] for m in self.results],
            'CV_Mean_R2': [self.results[m]['cv_mean'] for m in self.results],
            'CV_Std_R2': [self.results[m]['cv_std'] for m in self.results]
        })
        
        # Sort by Test R2
        comparison_df = comparison_df.sort_values('Test_R2', ascending=False)
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Identify best model
        self.best_model_name = comparison_df.iloc[0]['Model']
        self.best_model = self.results[self.best_model_name]['model']
        
        print("\n" + "="*80)
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print(f"   Test R²: {comparison_df.iloc[0]['Test_R2']:.4f}")
        print(f"   Test RMSE: {comparison_df.iloc[0]['Test_RMSE']:.4f}")
        print(f"   Test MAPE: {comparison_df.iloc[0]['Test_MAPE']:.2f}%")
        print("="*80)
        
        return comparison_df
    
    def visualize_results(self, save_path=None):
        """
        Visualize model results
        
        Parameters:
        -----------
        save_path : str or Path, optional
            Path to save figures
        """
        # Create a comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # 1. R² Score Comparison
        ax1 = plt.subplot(2, 3, 1)
        models = list(self.results.keys())
        train_r2 = [self.results[m]['train_metrics']['r2'] for m in models]
        test_r2 = [self.results[m]['test_metrics']['r2'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8)
        ax1.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Score Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RMSE Comparison
        ax2 = plt.subplot(2, 3, 2)
        train_rmse = [self.results[m]['train_metrics']['rmse'] for m in models]
        test_rmse = [self.results[m]['test_metrics']['rmse'] for m in models]
        
        ax2.bar(x - width/2, train_rmse, width, label='Train RMSE', alpha=0.8)
        ax2.bar(x + width/2, test_rmse, width, label='Test RMSE', alpha=0.8)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cross-Validation Scores
        ax3 = plt.subplot(2, 3, 3)
        cv_means = [self.results[m]['cv_mean'] for m in models]
        cv_stds = [self.results[m]['cv_std'] for m in models]
        
        ax3.barh(models, cv_means, xerr=cv_stds, alpha=0.8)
        ax3.set_xlabel('Cross-Validation R² Score')
        ax3.set_title('5-Fold Cross-Validation Results', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Best Model - Actual vs Predicted (Train)
        ax4 = plt.subplot(2, 3, 4)
        y_train_pred = self.results[self.best_model_name]['y_train_pred']
        ax4.scatter(self.y_train, y_train_pred, alpha=0.5)
        ax4.plot([self.y_train.min(), self.y_train.max()], 
                 [self.y_train.min(), self.y_train.max()], 
                 'r--', lw=2)
        ax4.set_xlabel('Actual Sales')
        ax4.set_ylabel('Predicted Sales')
        ax4.set_title(f'{self.best_model_name} - Training Set', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Best Model - Actual vs Predicted (Test)
        ax5 = plt.subplot(2, 3, 5)
        y_test_pred = self.results[self.best_model_name]['y_test_pred']
        ax5.scatter(self.y_test, y_test_pred, alpha=0.5, color='green')
        ax5.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 
                 'r--', lw=2)
        ax5.set_xlabel('Actual Sales')
        ax5.set_ylabel('Predicted Sales')
        ax5.set_title(f'{self.best_model_name} - Test Set', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Residual Plot
        ax6 = plt.subplot(2, 3, 6)
        residuals = self.y_test - y_test_pred
        ax6.scatter(y_test_pred, residuals, alpha=0.5, color='purple')
        ax6.axhline(y=0, color='r', linestyle='--', lw=2)
        ax6.set_xlabel('Predicted Sales')
        ax6.set_ylabel('Residuals')
        ax6.set_title(f'{self.best_model_name} - Residual Plot', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Model comparison plots saved to {save_path}")
        
        plt.show()
    
    def feature_importance_analysis(self, feature_names, save_path=None):
        """
        Analyze feature importance for tree-based models
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
        save_path : str or Path, optional
            Path to save figure
        """
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        tree_based_models = ['Decision Tree', 'Random Forest', 'Gradient Boosting']
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for idx, model_name in enumerate(tree_based_models):
            if model_name in self.results:
                model = self.results[model_name]['model']
                importance = model.feature_importances_
                
                # Create DataFrame for better visualization
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                print(f"\n{model_name} - Top 10 Important Features:")
                print("-" * 80)
                print(importance_df.head(10).to_string(index=False))
                
                # Plot
                axes[idx].barh(importance_df['Feature'][:10], 
                              importance_df['Importance'][:10])
                axes[idx].set_xlabel('Importance')
                axes[idx].set_title(f'{model_name}\nFeature Importance', 
                                   fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Feature importance plots saved to {save_path}")
        
        plt.show()
    
    def hyperparameter_tuning(self, model_name, param_grid, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Parameters:
        -----------
        model_name : str
            Name of the model to tune
        param_grid : dict
            Parameter grid for GridSearchCV
        cv : int
            Number of cross-validation folds
        """
        print("\n" + "="*80)
        print(f"HYPERPARAMETER TUNING: {model_name}")
        print("="*80)
        
        if model_name not in self.models:
            print(f"✗ Model {model_name} not found")
            return None
        
        base_model = self.models[model_name]
        
        print(f"\nParameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        print(f"\nPerforming Grid Search with {cv}-fold cross-validation...")
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, 
            scoring='r2', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Grid Search Complete!")
        print(f"\nBest Parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\nBest Cross-Validation R² Score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_test_pred = grid_search.best_estimator_.predict(self.X_test)
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        print(f"Test R² Score: {test_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        
        # Update results with tuned model
        self.results[f"{model_name} (Tuned)"] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'test_r2': test_r2,
            'test_rmse': test_rmse
        }
        
        return grid_search.best_estimator_
    
    def save_model(self, model_name, filepath):
        """
        Save trained model to disk
        
        Parameters:
        -----------
        model_name : str
            Name of the model to save
        filepath : str or Path
            Path to save the model
        """
        if model_name not in self.results:
            print(f"✗ Model {model_name} not found")
            return False
        
        try:
            model = self.results[model_name]['model']
            joblib.dump(model, filepath)
            print(f"✓ Model '{model_name}' saved to {filepath}")
            return True
        except Exception as e:
            print(f"✗ Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath):
        """
        Load a saved model from disk
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the saved model
        """
        try:
            model = joblib.load(filepath)
            print(f"✓ Model loaded from {filepath}")
            return model
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            return None
    
    def predict(self, X_new, model_name=None):
        """
        Make predictions using trained model
        
        Parameters:
        -----------
        X_new : array-like or DataFrame
            New data for prediction
        model_name : str, optional
            Specific model to use (uses best model if None)
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.results:
            print(f"✗ Model {model_name} not found")
            return None
        
        model = self.results[model_name]['model']
        predictions = model.predict(X_new)
        
        return predictions