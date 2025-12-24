# Pre-Seasonal Streamflow Forecasting

## Introduction
Accurate seasonal streamflow forecasting is essential for hydropower operations, reservoir regulation, ecological planning, and flood mitigation, especially in snow-dominated regions. The Michipicoten River watershed, draining into northeastern Lake Superior in Ontario, exhibits significant seasonal variability driven by snow accumulation, melt processes, and large-scale climate patterns. This project focuses on pre-seasonal (April–June, AMJ) streamflow forecasting using early-season (January–March, JFM) climate predictors.

## Project Questions
- Can Principal Component Regression (PCR) provide better prediction for AMJ streamflow than Multiple Linear Regression (MLR)?
- Which hydroclimatic variables (precipitation, temperature, SWE, potential evapotranspiration, mean sea-level pressure, and surface pressure) are most influential?
- How do PCR and MLR differ in performance metrics for AMJ streamflow prediction?

## Objectives
- Develop PCR and MLR models for predicting AMJ streamflow using JFM hydroclimatic predictors.  
- Identify and rank the most influential predictors.  
- Evaluate model performance using Pearson correlation (R), RMSE, MAE, Nash-Sutcliffe Efficiency (NSE), and RPSS.

## Methodology
The project uses a systematic approach:

1. **Data Preparation and Preprocessing**  
   - Hydroclimatic variables (Precipitation, Temperature, SWE, PET, MSLP, SP) obtained from ERA5 reanalysis and averaged for JFM.  
   - Observed AMJ streamflow (1991–2019) obtained from Environment and Climate Change Canada (station 02BD002).  
   - Variables standardized to zero mean and unit variance.

2. **Dimensionality Reduction**  
   - Principal Component Analysis (PCA) applied to reduce multicollinearity.  
   - Three principal components selected based on 80% cumulative variance explained.

3. **Model Development**  
   - **PCR**: Uses PCs as predictors for AMJ streamflow.  
   - **MLR**: Uses original predictors directly.

4. **Model Evaluation**  
   - Performance metrics: Pearson correlation (R), RMSE, MAE, NSE, RPSS.  
   - Scatter plots, contingency tables, and predictor influence ranking used for assessment.  

**Tools Used:** Python (data processing, PCA, PCR, visualization), QGIS (watershed delineation), Climate Predictability Tool (MLR modeling).

## Data Sources and Analysis
- **Study Area:** Michipicoten River watershed, Ontario, Canada.  
- **Predictors:** JFM hydroclimatic variables from ERA5 reanalysis: Precipitation (Pr), Temperature (T), Snow Water Equivalent (SWE), Potential Evapotranspiration (PET), Mean Sea-Level Pressure (MSLP), Surface Pressure (SP).  
- **Predictand:** Observed AMJ streamflow from Hydrometric Station 02BD002 (1991–2019).
- Heatmaps and scatter plots revealed multicollinearity, motivating the use of PCR.  
- Standardization ensured comparability across variables.

## Results
- **Variance Explained:** Three PCs captured >80% cumulative variance.
- Model analysis results shows SWE, precipitation, and temperature as most influential predictors.  
- **Model Performance:**

| Model | RMSE | MAE | Pearson R | NSE | RPSS |
|-------|------|-----|-----------|-----|------|
| PCR   | 25.88|22.01| 0.71      |0.51 |0.716 |
| MLR   | 30.72|25.59| 0.595     |0.31 | -    |

- **Predictor Influence (PCR):** SWE > Precipitation > Temperature > MSLP > SP > PET.  
- PCR outperformed MLR, especially in peak flow prediction.  
- Figures and tables are available in the `/figures` folder.

## Discussion
- PCR reduces multicollinearity and leverages principal components, providing better pre-seasonal forecasts.  
- SWE is the most critical variable due to its role in snowpack accumulation and spring melt.  
- MLR suffers from predictor multicollinearity, resulting in less stable forecasts.  

**Recommendations for Improvement:**  
- Include additional climate indices and higher-resolution datasets.  
- Integrate statistical and machine learning models.  
- Consider climate change impacts on streamflow prediction.  
- Compare with persistence and other predictive models.

## Project Structure / Files
- `/documents` – Contains all text documents, notes, and supplementary reports.  
- `/figures` – Plots, charts, PCA visualizations, and forecast comparison figures.  
- `streamflow_analysis.ipynb` – Jupyter Notebook with data processing, PCA, PCR, MLR, and visualization workflow.  
- `data/` – Processed datasets (ERA5 reanalysis and observed streamflow).  
- `README.md` – General overview of the project.  

## References
- Arnal, L. et al. (2024). *FROSTBYTE: A reproducible data-driven workflow for probabilistic seasonal streamflow forecasting in snow-fed river basins across North America*. Hydrology and Earth System Sciences, 28, 4127–4155.  
- Barnett, T. P., Adam, J. C., & Lettenmaier, D. P. (2005). *Potential impacts of a warming climate on water availability in snow-dominated regions*.  
- Fleming, S. W., & Garen, D. C. (2022). *Simplified Cross-Validation in Principal Component Regression (PCR) and PCR-Like Machine Learning for Water Supply Forecasting*. JAWRA, 58(4), 528–546.  
- Luce, C. H. et al. (2014). *Sensitivity of summer stream temperatures to climate variability in the Pacific Northwest*.  
- Risley, J. C. et al. (2005). *An Analysis of Statistical Methods for Seasonal Flow Forecasting in the Upper Klamath River Basin of Oregon and California*. U.S. Geological Survey Scientific Investigations Report 2005-5177.  
- Wuttichaikitcharoen, P., & Babel, M. S. (2014). *Principal component and multiple regression analyses for the estimation of suspended sediment yield in ungauged basins of Northern Thailand*.  
- Razavi, T., & Coulibaly, P. (2013). *Streamflow prediction in ungauged basins: Review of regionalization methods*. Journal of Hydrologic Engineering.
