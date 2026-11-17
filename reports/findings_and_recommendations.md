# Findings and Recommendations

- Sales rows: 421,570; stores: 45; departments: 81.
- Date range: 2010-02-05 to 2012-10-26.
- Negative sales preserved: 1,285.
- Markdown missingness: {"MarkDown1": 64.26, "MarkDown2": 73.61, "MarkDown3": 67.48, "MarkDown4": 67.98, "MarkDown5": 64.08}.
- Forecast MAE: 1,246.49; RMSE: 2,634.48; WAPE: 7.99%.
- Forecast holdout: 2012-08-10 to 2012-10-26.
- Anomaly candidates: 15,793.
- Silhouette scores: {"2": 0.30324070507429673, "3": 0.36396820423346365, "4": 0.3647526180640889, "5": 0.369272931636322, "6": 0.38378583928972027}; selected k=6.

Treat forecast metrics as baseline results from one holdout. Review anomaly candidates instead of deleting automatically. Segments are store-department profiles. Department sales co-movement is not proof of co-purchase.
