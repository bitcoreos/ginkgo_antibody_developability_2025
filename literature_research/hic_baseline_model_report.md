
# Baseline Model for Antibody Hydrophobicity (HIC) Prediction

## Dataset Analysis
- Dataset: GDPa1 v1.2 (246 antibodies)
- Target variable: HIC (Hydrophobicity Interaction Chromatography)
- Features used: PR_CHO, AC-SINS_pH7.4, Tm2, Titer

## Model Implementation
Two baseline models were implemented:
1. Linear Regression
2. Simple Neural Network (5 hidden units, 1000 max iterations)

## Model Performance

### Linear Regression
- R2 scores: [-0.14989576663003024, -0.013641805923011674, -0.024413005971327006, 0.02930381555649375, -0.0540890536495886]
- Mean R2: -0.0425
- RMSE scores: [np.float64(0.3063309715865701), np.float64(0.2819874791085107), np.float64(0.28801406010324565), np.float64(0.3658824193783904), np.float64(0.4330790142754181)]
- Mean RMSE: 0.3351

Feature Coefficients:
- PR_CHO: 0.1698
- AC-SINS_pH7.4: 0.0007
- Tm2: -0.0124
- Titer: -0.0001
- Intercept: 3.8426


### Neural Network
- R2 scores: [-3.4845815431944587, -3.2772363523318644, -3.107605797853177, -2.307819587397013, -2.5208954077328753]
- Mean R2: -2.9396
- RMSE scores: [np.float64(0.6049543910702776), np.float64(0.5792542017283382), np.float64(0.576727322570897), np.float64(0.6754151348750529), np.float64(0.7915075283320784)]
- Mean RMSE: 0.6456

## Findings and Limitations
1. Both baseline models performed poorly with negative R2 scores, indicating they perform worse than a simple mean predictor.
2. The features used (PR_CHO, AC-SINS_pH7.4, Tm2, Titer) have very low correlation with HIC.
3. According to competition guidelines, IgG subclass should be included as a feature, but this information was not found in the dataset.
4. Sequence-based features were not used in this baseline model but could potentially improve performance.

## Comparison with Existing Models
During this research, we discovered that more sophisticated models for HIC prediction already exist in the workspace:
- Several random forest and XGBoost models with 'delta_G_ML' features
- A final ridge regression model with 3 features

These models likely use more advanced features derived from molecular modeling or sequence analysis, which explains their better performance compared to our simple baseline.

## Recommendations
1. Include IgG subclass as a categorical feature if available in the dataset.
2. Consider using sequence-derived features to improve model performance.
3. Explore more sophisticated modeling approaches beyond linear regression and simple neural networks.
4. Investigate the features used in existing successful models (particularly those with 'delta_G_ML' features) for future work.
