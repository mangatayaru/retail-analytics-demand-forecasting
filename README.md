# Integrated Retail Analytics for Store Optimization and Demand Forecasting

## Overview
End-to-end capstone using the three supplied weekly retail datasets. Covers data validation, feature engineering, time-based anomaly detection, operational segmentation, demand forecasting, external-driver analysis, department sales co-movement, and practical recommendations.

## Data profile
- Sales rows: 421,570
- Stores: 45; departments: 81
- Date range: 2010-02-05 to 2012-10-26
- Negative sales rows preserved: 1,285
- Markdown missingness: {"MarkDown1": 64.26, "MarkDown2": 73.61, "MarkDown3": 67.48, "MarkDown4": 67.98, "MarkDown5": 64.08}

## Baseline results
- Forecast holdout: 2012-08-10 to 2012-10-26
- MAE: 1,246.49; RMSE: 2,634.48; WAPE: 7.99%
- Anomaly candidates: 15,793
- Silhouette by k: {"2": 0.30324070507429673, "3": 0.36396820423346365, "4": 0.3647526180640889, "5": 0.369272931636322, "6": 0.38378583928972027}; highest tested score at k=6

Results use one chronological holdout; compare with seasonal-naive baselines and rolling-origin backtesting before deployment.

## Scope limitation
No customer IDs, product SKUs or transaction baskets are present. The project performs **store-department operational segmentation**, not customer-level segmentation. Department sales co-movement is only a hypothesis generator, not true market basket analysis.

## Structure
- `data/`: supplied CSVs
- `notebooks/retail_analytics_capstone.ipynb`: detailed notebook
- `src/retail_pipeline.py`, `src/run_pipeline.py`: reusable pipeline
- `reports/`: metrics, predictions, anomaly flags, segments, figures
- `models/`: fitted segmentation and forecasting artifacts
- `docs/methodology_report.docx`: Google Docs-ready report
- `docs/methodology_report.md`: methodology
- `docs/video_presentation_script.md`: timed narration script

## Run
From the repository root, install packages from `requirements.txt`, then run `python src/run_pipeline.py`. Open the notebook in Jupyter or VS Code and execute it from top to bottom.

## Recommendations
1. Combine forecasts with lead times, supplier constraints and service-level targets.
2. Investigate anomaly candidates with holiday/markdown context; never delete automatically.
3. Tailor monitoring and safety stock to demand volatility.
4. Validate markdown and cross-merchandising strategies with controlled experiments.
5. Obtain transaction-level data before true market basket analysis or customer personalization.

