"""Retail analytics pipeline. Run from repository root: python src/run_pipeline.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS, FIGURES, MODELS = ROOT/"data", ROOT/"reports", ROOT/"reports"/"figures", ROOT/"models"
for p in [REPORTS, FIGURES, MODELS]:
    p.mkdir(parents=True, exist_ok=True)

def load_prepare():
    sales = pd.read_csv(DATA/"sales data-set.csv")
    stores = pd.read_csv(DATA/"stores data-set.csv")
    features = pd.read_csv(DATA/"Features data set.csv")
    for x in [sales, features]:
        x["Date"] = pd.to_datetime(x["Date"], dayfirst=True, errors="coerce")
        x["IsHoliday"] = x["IsHoliday"].astype(str).str.lower().map({"true": True, "false": False})
    sales = sales.dropna(subset=["Date","Store","Dept","Weekly_Sales"]).drop_duplicates(["Store","Dept","Date"])
    features = features.dropna(subset=["Date","Store"]).drop_duplicates(["Store","Date"])
    df = sales.merge(stores, on="Store", how="left", validate="many_to_one")
    df = df.merge(features, on=["Store","Date"], how="left", suffixes=("", "_feature"), validate="many_to_one")
    df["Year"] = df.Date.dt.year
    df["Month"] = df.Date.dt.month
    df["WeekOfYear"] = df.Date.dt.isocalendar().week.astype(int)
    df["Quarter"] = df.Date.dt.quarter
    df["YearEnd"] = df.Month.isin([11,12]).astype(int)
    df["IsHoliday"] = df["IsHoliday"].fillna(df["IsHoliday_feature"]).fillna(False).astype(bool)
    for c in [c for c in df.columns if c.startswith("MarkDown") and not c.endswith("_was_missing")]:
        df[c+"_was_missing"] = df[c].isna().astype(int)
        df[c] = df[c].fillna(0)
    return df.sort_values(["Store","Dept","Date"]).reset_index(drop=True)

def add_anomalies(df):
    g = df.groupby(["Store","Dept"])["Weekly_Sales"]
    df["rolling_median_13"] = g.transform(lambda s: s.shift(1).rolling(13,min_periods=6).median())
    df["rolling_mad_13"] = g.transform(lambda s: (s.shift(1)-s.shift(1).rolling(13,min_periods=6).median()).abs().rolling(13,min_periods=6).median())
    df["robust_z"] = (df["Weekly_Sales"]-df["rolling_median_13"])/(1.4826*df["rolling_mad_13"].replace(0,np.nan))
    df["anomaly_flag"] = df["robust_z"].abs().ge(4.5).fillna(False)
    return df

def main():
    df = add_anomalies(load_prepare())
    df.loc[df.anomaly_flag].to_csv(REPORTS/"anomaly_flags.csv", index=False)
    # Store-department operational segments, not customer segments.
    seg = df.groupby(["Store","Dept","Type","Size"],dropna=False).agg(
        mean_sales=("Weekly_Sales","mean"), median_sales=("Weekly_Sales","median"),
        sales_std=("Weekly_Sales","std"), sales_cv=("Weekly_Sales",lambda s:s.std()/(abs(s.mean())+1e-6)),
        markdown_mean=("MarkDown1","mean"), markdown4_mean=("MarkDown4","mean"),
        negative_sales_rate=("Weekly_Sales",lambda s:(s<0).mean()), n_weeks=("Weekly_Sales","size")
    ).reset_index()
    holiday = df.groupby(["Store","Dept"]).apply(lambda x:x.loc[x.IsHoliday,"Weekly_Sales"].mean() if x.IsHoliday.any() else np.nan, include_groups=False).rename("holiday_mean").reset_index()
    seg = seg.merge(holiday,on=["Store","Dept"],how="left")
    seg["holiday_lift"] = seg.holiday_mean/seg.mean_sales.replace(0,np.nan)
    features = ["mean_sales","median_sales","sales_std","sales_cv","markdown_mean","markdown4_mean","negative_sales_rate","holiday_lift","Size"]
    X = seg[features].replace([np.inf,-np.inf],np.nan)
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    Xs = sc.fit_transform(imp.fit_transform(X))
    sil = {}
    for k in range(2,7):
        lab = KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(Xs)
        sil[k] = float(silhouette_score(Xs,lab,sample_size=min(10000,len(Xs)),random_state=42))
    k = max(sil,key=sil.get)
    km = KMeans(n_clusters=k,random_state=42,n_init=10).fit(Xs)
    seg["Segment"] = km.labels_
    seg.to_csv(REPORTS/"segment_profiles.csv",index=False)
    joblib.dump({"imputer":imp,"scaler":sc,"model":km,"features":features},MODELS/"store_department_segmentation.joblib")
    # Forecast lags and rolling features, always shifted to prevent target leakage.
    for lag in [1,2,4,13,52]:
        df[f"lag_{lag}"] = df.groupby(["Store","Dept"]).Weekly_Sales.shift(lag)
    for w in [4,13]:
        df[f"roll_mean_{w}"] = df.groupby(["Store","Dept"]).Weekly_Sales.transform(lambda s:s.shift(1).rolling(w,min_periods=2).mean())
        df[f"roll_std_{w}"] = df.groupby(["Store","Dept"]).Weekly_Sales.transform(lambda s:s.shift(1).rolling(w,min_periods=2).std())
    data = df.dropna(subset=["lag_1","lag_2"]).copy()
    cutoff = data.Date.max()-pd.Timedelta(weeks=12)
    train, test = data[data.Date<=cutoff], data[data.Date>cutoff]
    nums = ["Store","Dept","Size","Temperature","Fuel_Price","CPI","Unemployment","Year","Month","WeekOfYear","Quarter","YearEnd","IsHoliday",
            "MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5","lag_1","lag_2","lag_4","lag_13","lag_52","roll_mean_4","roll_std_4","roll_mean_13","roll_std_13"]
    nums = [c for c in nums if c in data.columns]
    xt = pd.get_dummies(train[nums+["Type"]],columns=["Type"],dummy_na=True)
    xv = pd.get_dummies(test[nums+["Type"]],columns=["Type"],dummy_na=True).reindex(columns=xt.columns,fill_value=0)
    model = Pipeline([("imputer",SimpleImputer(strategy="median")),("regressor",HistGradientBoostingRegressor(max_iter=120,learning_rate=.08,max_leaf_nodes=31,l2_regularization=1,random_state=42))])
    model.fit(xt,train.Weekly_Sales)
    pred = model.predict(xv)
    mae = float(mean_absolute_error(test.Weekly_Sales,pred))
    rmse = float(mean_squared_error(test.Weekly_Sales,pred)**.5)
    wape = float(np.abs(test.Weekly_Sales.to_numpy()-pred).sum()/np.abs(test.Weekly_Sales.to_numpy()).sum())
    out = test[["Store","Dept","Date","Weekly_Sales"]].copy()
    out["Predicted_Sales"], out["Absolute_Error"] = pred, np.abs(test.Weekly_Sales.to_numpy()-pred)
    out.to_csv(REPORTS/"forecast_predictions.csv",index=False)
    joblib.dump({"model":model,"columns":list(xt.columns),"numeric_features":nums},MODELS/"weekly_sales_forecaster.joblib")
    # Co-movement proxy only: cannot replace transaction-level market basket analysis.
    pv = df.pivot_table(index=["Store","Date"],columns="Dept",values="Weekly_Sales",aggfunc="sum")
    rows=[]
    for store, sub in pv.groupby(level="Store"):
        corr=sub.droplevel("Store").corr(min_periods=20)
        cs=list(corr.columns)
        for i,a in enumerate(cs):
            for b in cs[i+1:]:
                if pd.notna(corr.loc[a,b]): rows.append((int(store),int(a),int(b),float(corr.loc[a,b])))
    pd.DataFrame(rows,columns=["Store","Dept_A","Dept_B","Sales_Correlation"]).sort_values("Sales_Correlation",ascending=False).head(1000).to_csv(REPORTS/"department_sales_comovement_proxy_top1000.csv",index=False)
    metrics={"forecast":{"cutoff":str(cutoff.date()),"test_start":str(test.Date.min().date()),"test_end":str(test.Date.max().date()),"n_train":len(train),"n_test":len(test),"MAE":mae,"RMSE":rmse,"WAPE":wape},"silhouette_by_k":sil,"selected_k":k,"anomaly_candidates":int(df.anomaly_flag.sum())}
    (REPORTS/"model_metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
if __name__ == "__main__": main()
