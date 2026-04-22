"""
Main Execution Script for Sales Prediction Project
IBM Data Science Standards

Author: Your Name
Date: 2024
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import SalesPredictionModel
import config

# Set plotting style
sns.set_style(config.STYLE)
plt.rcParams['figure.figsize'] = config.FIGURE_SIZE

def main():
    """
    Main execution function
    """
    print("\n" + "="*80)
    print("SALES PREDICTION PROJECT")
    print("IBM Data Science Standards")
    print("="*80)
    
    # ========================================================================
    # PHASE 1: DATA LOADING AND EXPLORATION
    # ========================================================================
    print("\n" + "#"*80)
    print("# PHASE 1: DATA LOADING AND EXPLORATION")
    print("#"*80)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config.RAW_DATA_FILE)
    
    # Load data
    df = preprocessor.load_data()
    
    if df is None:
        print("\n✗ Please download the dataset from Kaggle and place it in:")
        print(f"   {config.RAW_DATA_DIR}")
        print("\nDataset URL: https://www.kaggle.com/datasets/bumba5341/advertisingcsv")
        return
    
    # Initial exploration
    exploration_report = preprocessor.initial_exploration()
    
    # Detect outliers
    outliers_info = preprocessor.detect_outliers(method='iqr', threshold=1.5)
    
    # Visualize distributions
    preprocessor.visualize_distributions(
        save_path=config.FIGURES_DIR / 'data_distributions.png'
    )
    
    # ========================================================================
    # PHASE 2: DATA CLEANING
    # ========================================================================
    print("\n" + "#"*80)
    print("# PHASE 2: DATA CLEANING")
    print("#"*80)
    
    # Clean data
    df_clean = preprocessor.clean_data(
        remove_duplicates=True,
        handle_missing='drop',
        handle_outliers='keep',  # Keep outliers as they might be valid
        outlier_method='iqr'
    )
    
    # Save cleaned data
    preprocessor.save_processed_data(config.PROCESSED_DATA_FILE)
    
    # ========================================================================
    # PHASE 3: FEATURE ENGINEERING
    # ========================================================================
    print("\n" + "#"*80)
    print("# PHASE 3: FEATURE ENGINEERING")
    print("#"*80)
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(df_clean)
    
    # Define base features
    base_features = config.FEATURE_COLUMNS
    target = config.TARGET_VARIABLE
    
    # Create interaction features
    interaction_features = feature_engineer.create_interaction_features(base_features)
    
    # Create polynomial features
    polynomial_features = feature_engineer.create_polynomial_features(
        base_features, degree=2
    )
    
    # Create ratio features
    ratio_features = feature_engineer.create_ratio_features(base_features)
    
    # Create aggregate features
    aggregate_features = feature_engineer.create_aggregate_features(base_features)
    
    # Get engineered dataset
    df_engineered = feature_engineer.get_engineered_data()
    
    print(f"\n✓ Feature Engineering Complete!")
    print(f"  Original features: {len(base_features)}")
    print(f"  Total features created: {len(df_engineered.columns) - len(df_clean.columns)}")
    print(f"  Final dataset shape: {df_engineered.shape}")
    
    # Correlation analysis
    target_corr = feature_engineer.analyze_correlations(
        target_col=target,
        save_path=config.FIGURES_DIR / 'correlation_matrix.png'
    )
    
    # Feature selection based on correlation
    selected_features = feature_engineer.select_features(
        target_col=target,
        method='correlation',
        threshold=0.05,  # Select features with >5% correlation
        top_n=20  # Or select top 20 features
    )
    
    # ========================================================================
    # PHASE 4: MODEL BUILDING AND EVALUATION
    # ========================================================================
    print("\n" + "#"*80)
    print("# PHASE 4: MODEL BUILDING AND EVALUATION")
    print("#"*80)
    
    # Initialize model class
    sales_model = SalesPredictionModel(random_state=config.RANDOM_STATE)
    
    # Prepare data with selected features
    X_train, X_test, y_train, y_test = sales_model.prepare_data(
        dataframe=df_engineered,
        target_col=target,
        feature_cols=selected_features,
        test_size=config.TEST_SIZE
    )
    
    # Train all models
    sales_model.train_models()
    
    # Compare models
    comparison_df = sales_model.compare_models()
    
    # Save comparison results
    comparison_df.to_csv(
        config.REPORTS_DIR / 'model_comparison.csv', 
        index=False
    )
    
    # Visualize results
    sales_model.visualize_results(
        save_path=config.FIGURES_DIR / 'model_comparison.png'
    )
    
    # Feature importance analysis
    sales_model.feature_importance_analysis(
        feature_names=selected_features,
        save_path=config.FIGURES_DIR / 'feature_importance.png'
    )
    
    # ========================================================================
    # PHASE 5: HYPERPARAMETER TUNING
    # ========================================================================
    print("\n" + "#"*80)
    print("# PHASE 5: HYPERPARAMETER TUNING (BEST MODEL)")
    print("#"*80)
    
    # Define parameter grids for top models
    
    # For Random Forest
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # For Gradient Boosting
    gb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }
    
    # Tune the best performing model
    best_model_name = sales_model.best_model_name
    
    if 'Random Forest' in best_model_name:
        tuned_model = sales_model.hyperparameter_tuning(
            model_name='Random Forest',
            param_grid=rf_param_grid,
            cv=5
        )
    elif 'Gradient Boosting' in best_model_name:
        tuned_model = sales_model.hyperparameter_tuning(
            model_name='Gradient Boosting',
            param_grid=gb_param_grid,
            cv=5
        )
    
    # ========================================================================
    # PHASE 6: SAVE BEST MODEL
    # ========================================================================
    print("\n" + "#"*80)
    print("# PHASE 6: SAVING BEST MODEL")
    print("#"*80)
    
    # Save the best model
    model_filename = f"best_model_{best_model_name.replace(' ', '_').lower()}.pkl"
    sales_model.save_model(
        model_name=best_model_name,
        filepath=config.MODELS_DIR / model_filename
    )
    
    # ========================================================================
    # PHASE 7: BUSINESS INSIGHTS AND RECOMMENDATIONS
    # ========================================================================
    print("\n" + "#"*80)
    print("# PHASE 7: BUSINESS INSIGHTS AND RECOMMENDATIONS")
    print("#"*80)
    
    # Analyze advertising impact
    print("\nAdvertising Channel Impact Analysis:")
    print("="*80)
    
    # Get feature importances or coefficients
    if hasattr(sales_model.best_model, 'feature_importances_'):
        importance = sales_model.best_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance_df.head(10).to_string(index=False))
        
    elif hasattr(sales_model.best_model, 'coef_'):
        coef = sales_model.best_model.coef_
        feature_coef_df = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': coef
        }).sort_values('Coefficient', ascending=False, key=abs)
        
        print("\nTop 10 Most Influential Features:")
        print(feature_coef_df.head(10).to_string(index=False))
    
    # Calculate ROI for each advertising channel
    print("\n\nAdvertising Channel ROI Analysis:")
    print("="*80)
    
    for channel in base_features:
        if channel in selected_features:
            # Calculate average spending and sales
            avg_spend = df_clean[channel].mean()
            avg_sales = df_clean[target].mean()
            
            # Calculate correlation with sales
            correlation = df_clean[channel].corr(df_clean[target])
            
            print(f"\n{channel}:")
            print(f"  Average Spend: ${avg_spend:.2f}")
            print(f"  Correlation with Sales: {correlation:.4f}")
            print(f"  Estimated ROI: {(correlation * avg_sales / avg_spend):.2f}x")
    
    # Generate actionable insights
    print("\n\n" + "="*80)
    print("ACTIONABLE BUSINESS INSIGHTS")
    print("="*80)
    
    insights = f"""
1. MODEL PERFORMANCE:
   - Best Model: {best_model_name}
   - Prediction Accuracy (R²): {comparison_df.iloc[0]['Test_R2']:.4f}
   - Average Prediction Error (RMSE): ${comparison_df.iloc[0]['Test_RMSE']:.2f}
   - This means the model can explain {comparison_df.iloc[0]['Test_R2']*100:.1f}% 
     of the variance in sales.

2. KEY FINDINGS:
   {generate_key_findings(df_clean, base_features, target)}

3. RECOMMENDATIONS:
   {generate_recommendations(df_clean, base_features, target)}

4. OPTIMIZATION STRATEGIES:
   - Use the model to predict sales for different advertising budget allocations
   - Test scenarios: What if we increase TV budget by 20%?
   - Identify diminishing returns threshold for each channel
   - Optimize budget allocation for maximum ROI

5. RISK FACTORS:
   - Model assumes current market conditions remain stable
   - External factors (economy, competition) not included in current model
   - Recommend quarterly model retraining with new data
   - Consider A/B testing for validation

6. NEXT STEPS:
   - Deploy model to production environment
   - Set up automated prediction pipeline
   - Create dashboard for marketing team
   - Implement continuous monitoring and model updates
    """
    
    print(insights)
    
    # Save insights to file
    with open(config.REPORTS_DIR / 'business_insights.txt', 'w') as f:
        f.write(insights)
    
    print("\n✓ Business insights saved to:", config.REPORTS_DIR / 'business_insights.txt')
    
    # ========================================================================
    # PHASE 8: DEMONSTRATION - PREDICTION ON NEW DATA
    # ========================================================================
    print("\n" + "#"*80)
    print("# PHASE 8: DEMONSTRATION - SALES PREDICTION")
    print("#"*80)
    
    # Create sample scenarios
    print("\nPredicting sales for different advertising scenarios:")
    print("="*80)
    
    scenarios = pd.DataFrame({
        'Scenario': ['Low Budget', 'Medium Budget', 'High Budget', 'TV Focus', 
                     'Radio Focus', 'Balanced'],
        'TV': [50, 150, 250, 300, 100, 150],
        'Radio': [10, 25, 40, 20, 50, 25],
        'Newspaper': [5, 15, 30, 10, 10, 20]
    })
    
    # Engineer features for scenarios
    scenario_engineer = FeatureEngineer(scenarios)
    scenario_engineer.create_interaction_features(base_features)
    scenario_engineer.create_polynomial_features(base_features, degree=2)
    scenario_engineer.create_ratio_features(base_features)
    scenario_engineer.create_aggregate_features(base_features)
    
    scenarios_engineered = scenario_engineer.get_engineered_data()
    
    # Make predictions
    X_scenarios = scenarios_engineered[selected_features]
    predictions = sales_model.predict(X_scenarios)
    
    # Display results
    results_df = scenarios.copy()
    results_df['Predicted_Sales'] = predictions
    results_df['Total_Spend'] = results_df[['TV', 'Radio', 'Newspaper']].sum(axis=1)
    results_df['ROI'] = results_df['Predicted_Sales'] / results_df['Total_Spend']
    
    print("\n" + results_df.to_string(index=False))
    
    # Visualize scenario analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Predicted sales by scenario
    axes[0].bar(results_df['Scenario'], results_df['Predicted_Sales'], 
                color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('Predicted Sales ($)')
    axes[0].set_title('Predicted Sales by Advertising Scenario', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # ROI by scenario
    axes[1].bar(results_df['Scenario'], results_df['ROI'], 
                color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('ROI (Sales per $ Spent)')
    axes[1].set_title('Return on Investment by Scenario', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'scenario_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # PROJECT SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("PROJECT COMPLETION SUMMARY")
    print("="*80)
    
    summary = f"""
✓ Data loaded and explored: {len(df)} records
✓ Data cleaned and preprocessed
✓ Features engineered: {len(df_engineered.columns)} total features
✓ Models trained and evaluated: {len(sales_model.results)} models
✓ Best model identified: {best_model_name}
✓ Model saved to: {config.MODELS_DIR / model_filename}
✓ Reports generated in: {config.REPORTS_DIR}
✓ Visualizations saved in: {config.FIGURES_DIR}

Final Model Performance:
- R² Score: {comparison_df.iloc[0]['Test_R2']:.4f}
- RMSE: ${comparison_df.iloc[0]['Test_RMSE']:.2f}
- MAE: ${comparison_df.iloc[0]['Test_MAE']:.2f}
- MAPE: {comparison_df.iloc[0]['Test_MAPE']:.2f}%

The model is ready for deployment and can predict sales based on 
advertising spend across TV, Radio, and Newspaper channels.
    """
    
    print(summary)
    
    print("\n" + "="*80)
    print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")

def generate_key_findings(df, features, target):
    """Generate key findings from data analysis"""
    findings = []
    
    for feature in features:
        corr = df[feature].corr(df[target])
        avg_spend = df[feature].mean()
        findings.append(f"   - {feature}: Correlation = {corr:.3f}, "
                       f"Avg Spend = ${avg_spend:.2f}")
    
    return '\n'.join(findings)

def generate_recommendations(df, features, target):
    """Generate business recommendations"""
    recommendations = []
    
    # Find most impactful channel
    correlations = {feature: df[feature].corr(df[target]) for feature in features}
    best_channel = max(correlations, key=correlations.get)
    
    recommendations.append(f"   - Prioritize {best_channel} advertising "
                          f"(highest correlation: {correlations[best_channel]:.3f})")
    
    # Efficiency analysis
    for feature in features:
        efficiency = df[target].mean() / df[feature].mean()
        recommendations.append(f"   - {feature} efficiency: "
                              f"${efficiency:.2f} sales per $1 spent")
    
    return '\n'.join(recommendations)

if __name__ == "__main__":
    main()