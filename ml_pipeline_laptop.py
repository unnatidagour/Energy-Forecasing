

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings, json, os
warnings.filterwarnings('ignore')

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_data(n_hours=8760):
    dates = pd.date_range("2023-01-01", periods=n_hours, freq="h")
    df = pd.DataFrame({"datetime": dates})

    h   = df["datetime"].dt.hour
    dow = df["datetime"].dt.dayofweek
    doy = df["datetime"].dt.dayofyear
    mon = df["datetime"].dt.month

    base    = 200
    hourly  = 30*np.sin(2*np.pi*(h-6)/24) + 20*np.sin(2*np.pi*(h-18)/12)
    weekly  = np.where(dow < 5, 40, -20)
    annual  = 50*np.cos(2*np.pi*(doy-180)/365)
    temp    = 15 + 12*np.sin(2*np.pi*(doy-80)/365) + 5*np.random.randn(n_hours)
    t_eff   = np.where(temp>25,(temp-25)*4, np.where(temp<10,(10-temp)*5,0))
    noise   = np.random.normal(0, 10, n_hours)

    df["demand_kwh"]       = np.clip(base+hourly+weekly+annual+t_eff+noise, 50, 600)
    df["temperature"]      = temp
    df["hour"]             = h
    df["day_of_week"]      = dow
    df["month"]            = mon
    df["day_of_year"]      = doy
    df["is_weekend"]       = (dow >= 5).astype(int)
    df["is_peak"]          = ((h.between(6,9)) | (h.between(17,21))).astype(int)
    df["hour_sin"]         = np.sin(2*np.pi*h/24)
    df["hour_cos"]         = np.cos(2*np.pi*h/24)
    df["month_sin"]        = np.sin(2*np.pi*mon/12)
    df["month_cos"]        = np.cos(2*np.pi*mon/12)
    df["dow_sin"]          = np.sin(2*np.pi*dow/7)
    df["dow_cos"]          = np.cos(2*np.pi*dow/7)
    df["lag_1h"]           = df["demand_kwh"].shift(1)
    df["lag_24h"]          = df["demand_kwh"].shift(24)
    df["lag_168h"]         = df["demand_kwh"].shift(168)
    df["rolling_mean_24h"] = df["demand_kwh"].rolling(24).mean()
    return df.dropna().reset_index(drop=True)

FEATURES = [
    "temperature","hour_sin","hour_cos","month_sin","month_cos",
    "dow_sin","dow_cos","is_weekend","is_peak",
    "lag_1h","lag_24h","lag_168h","rolling_mean_24h"
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. STYLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
BG, FG, GRID = "#0F172A", "#F1F5F9", "#1E293B"
C1, C2, C3   = "#00E5C3", "#FF6B6B", "#3D9FFF"

def styled_fig(*args, **kw):
    fig = plt.figure(*args, **kw, facecolor=BG)
    return fig

def style(ax, title=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=9)
    for s in ax.spines.values(): s.set_edgecolor(GRID)
    ax.grid(color=GRID, lw=0.6, ls="--", alpha=0.7)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    if title: ax.set_title(title, color=FG, fontsize=11, fontweight="bold", pad=10)

# ─────────────────────────────────────────────────────────────────────────────
# 3. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_overview(df):
    fig, axes = plt.subplots(3,1, figsize=(14,10), facecolor=BG)
    fig.suptitle("Energy Demand — Dataset Overview", color=FG, fontsize=14, fontweight="bold")

    # Daily average
    ax = axes[0]
    daily = df.set_index("datetime")["demand_kwh"].resample("D").mean()
    ax.fill_between(daily.index, daily.values, alpha=0.35, color=C1)
    ax.plot(daily.index, daily.values, color=C1, lw=1.2)
    style(ax, "Daily Average Demand (Full Year)")
    ax.set_ylabel("kWh", color=FG)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # Hourly profile
    ax = axes[1]
    for label, mask, color in [("Weekday", df["is_weekend"]==0, C1),
                                ("Weekend", df["is_weekend"]==1, C2)]:
        p = df[mask].groupby("hour")["demand_kwh"].mean()
        ax.plot(p.index, p.values, color=color, lw=2, label=label)
    ax.legend(facecolor=GRID, labelcolor=FG)
    style(ax, "Average Hourly Profile: Weekday vs Weekend")
    ax.set_xlabel("Hour of Day", color=FG); ax.set_ylabel("kWh", color=FG)

    # Monthly box
    ax = axes[2]
    mnths = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    data  = [df[df["month"]==m]["demand_kwh"].values for m in range(1,13)]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color=C2, lw=2),
                    whiskerprops=dict(color=FG), capprops=dict(color=FG),
                    flierprops=dict(marker=".", color=C1, ms=2))
    for p in bp["boxes"]: p.set_facecolor(C1); p.set_alpha(0.4)
    ax.set_xticklabels(mnths)
    style(ax, "Monthly Demand Distribution")
    ax.set_ylabel("kWh", color=FG)

    plt.tight_layout()
    plt.savefig(f"{OUT}/01_demand_overview.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ 01_demand_overview.png")

def plot_regression_diagnostics(y_test, y_pred, coef_df):
    fig, axes = plt.subplots(2,2, figsize=(13,9), facecolor=BG)
    fig.suptitle("Linear Regression — Diagnostics", color=FG, fontsize=13, fontweight="bold")

    # 1. Actual vs predicted scatter
    ax = axes[0,0]
    ax.scatter(y_test, y_pred, s=4, alpha=0.4, color=C1)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, color=C2, lw=1.5, ls="--", label="Perfect fit")
    ax.legend(facecolor=GRID, labelcolor=FG)
    style(ax, "Actual vs Predicted")
    ax.set_xlabel("Actual kWh", color=FG); ax.set_ylabel("Predicted kWh", color=FG)

    # 2. Residuals
    resid = y_test - y_pred
    ax = axes[0,1]
    ax.scatter(y_pred, resid, s=4, alpha=0.35, color=C3)
    ax.axhline(0, color=C2, lw=1.5, ls="--")
    style(ax, "Residuals vs Fitted")
    ax.set_xlabel("Fitted Values", color=FG); ax.set_ylabel("Residuals", color=FG)

    # 3. Residual histogram
    ax = axes[1,0]
    ax.hist(resid, bins=60, color=C1, alpha=0.7, edgecolor=BG)
    style(ax, "Residual Distribution")
    ax.set_xlabel("Residual (kWh)", color=FG); ax.set_ylabel("Frequency", color=FG)

    # 4. Coefficient bar
    ax = axes[1,1]
    coef_df_s = coef_df.reindex(coef_df["coef"].abs().sort_values().index)
    colors = [C1 if v >= 0 else C2 for v in coef_df_s["coef"]]
    ax.barh(coef_df_s["feature"], coef_df_s["coef"], color=colors)
    style(ax, "Feature Coefficients")
    ax.set_xlabel("Coefficient Value", color=FG)

    plt.tight_layout()
    plt.savefig(f"{OUT}/02_regression_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ 02_regression_diagnostics.png")

def plot_actual_vs_predicted(df_test):
    fig, ax = plt.subplots(figsize=(14,5), facecolor=BG)
    n = 168
    ax.plot(range(n), df_test["actual"].values[:n], color=FG, lw=1.5, label="Actual")
    ax.plot(range(n), df_test["predicted"].values[:n], color=C1, lw=1.5, ls="--", label="Predicted (LR)")
    ax.fill_between(range(n), df_test["actual"].values[:n], df_test["predicted"].values[:n],
                    alpha=0.15, color=C2)
    ax.legend(facecolor=GRID, labelcolor=FG)
    style(ax, "Actual vs Predicted — 7-Day Snapshot (168 hours)")
    ax.set_xlabel("Hours", color=FG); ax.set_ylabel("kWh", color=FG)
    plt.tight_layout()
    plt.savefig(f"{OUT}/03_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ 03_actual_vs_predicted.png")

def plot_peak_analysis(df):
    fig, axes = plt.subplots(1,2, figsize=(13,5), facecolor=BG)
    fig.suptitle("Peak vs Off-Peak Analysis", color=FG, fontsize=13, fontweight="bold")

    # Heatmap
    ax = axes[0]
    pivot = df.groupby(["month","hour"])["demand_kwh"].mean().unstack()
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", origin="lower", extent=[0,23,1,12])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors=FG); cbar.set_label("kWh", color=FG)
    ax.set_xlabel("Hour", color=FG); ax.set_ylabel("Month", color=FG)
    style(ax, "Average Demand Heatmap (Month × Hour)")

    # Violin
    ax = axes[1]
    vp = ax.violinplot([df[df["is_peak"]==0]["demand_kwh"],
                        df[df["is_peak"]==1]["demand_kwh"]],
                       positions=[0,1], showmedians=True)
    for body, color in zip(vp["bodies"], [C1, C2]):
        body.set_facecolor(color); body.set_alpha(0.55)
    vp["cmedians"].set_color(FG)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Off-Peak","Peak"])
    style(ax, "Demand: Peak vs Off-Peak")
    ax.set_ylabel("kWh", color=FG)
    plt.tight_layout()
    plt.savefig(f"{OUT}/04_peak_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ 04_peak_analysis.png")

def plot_forecast(model, scaler, df):
    last = df.tail(170).copy().reset_index(drop=True)
    forecasts = []
    for i in range(168):
        prev_dt  = last["datetime"].iloc[-1]
        next_dt  = prev_dt + pd.Timedelta(hours=1)
        h  = next_dt.hour; dow = next_dt.dayofweek
        mon = next_dt.month; doy = next_dt.day_of_year
        new = {
            "datetime": next_dt, "month": mon,
            "temperature": 15+12*np.sin(2*np.pi*(doy-80)/365)+np.random.randn()*3,
            "is_weekend": int(dow>=5),
            "is_peak": int((6<=h<=9)or(17<=h<=21)),
            "hour_sin": np.sin(2*np.pi*h/24), "hour_cos": np.cos(2*np.pi*h/24),
            "month_sin": np.sin(2*np.pi*mon/12), "month_cos": np.cos(2*np.pi*mon/12),
            "dow_sin": np.sin(2*np.pi*dow/7), "dow_cos": np.cos(2*np.pi*dow/7),
            "lag_1h":  last["demand_kwh"].iloc[-1],
            "lag_24h": last["demand_kwh"].iloc[-24] if len(last)>=24 else last["demand_kwh"].mean(),
            "lag_168h":last["demand_kwh"].iloc[-168] if len(last)>=168 else last["demand_kwh"].mean(),
            "rolling_mean_24h": last["demand_kwh"].tail(24).mean(),
        }
        Xn = scaler.transform(pd.DataFrame([new])[FEATURES])
        pred = model.predict(Xn)[0]
        new["demand_kwh"] = pred
        last = pd.concat([last, pd.DataFrame([new])], ignore_index=True)
        forecasts.append((next_dt, pred))

    fcast = pd.DataFrame(forecasts, columns=["datetime","forecast_kwh"])

    fig, ax = plt.subplots(figsize=(14,5), facecolor=BG)
    hist = df.tail(72).set_index("datetime")["demand_kwh"]
    ax.plot(hist.index, hist.values, color=FG, lw=1.5, label="Historical (last 3 days)")
    ax.plot(fcast["datetime"], fcast["forecast_kwh"], color=C1, lw=1.8, ls="--", label="7-Day Forecast")
    ax.fill_between(fcast["datetime"], fcast["forecast_kwh"]*0.90, fcast["forecast_kwh"]*1.10,
                    alpha=0.18, color=C1, label="±10% Confidence Band")
    ax.axvline(df["datetime"].iloc[-1], color=C2, lw=1, ls=":")
    ax.legend(facecolor=GRID, labelcolor=FG)
    style(ax, "7-Day Energy Demand Forecast — Linear Regression")
    ax.set_ylabel("kWh", color=FG); ax.set_xlabel("Date", color=FG)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.tight_layout()
    plt.savefig(f"{OUT}/05_future_forecast.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ 05_future_forecast.png")
    return fcast

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  ENERGY DEMAND FORECASTING — LINEAR REGRESSION")
    print("═"*60)

    print("\n[1/5] Generating synthetic dataset...")
    df = generate_data(8760)
    df.to_csv(f"{OUT}/energy_dataset.csv", index=False)
    print(f"  Rows: {len(df)}  |  {df['datetime'].min().date()} → {df['datetime'].max().date()}")

    print("\n[2/5] Training Linear Regression model...")
    X = df[FEATURES]; y = df["demand_kwh"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test.values - y_pred)/y_test.values))*100

    print(f"\n  ── Model Performance ──────────────────")
    print(f"  MAE   : {mae:.2f} kWh")
    print(f"  RMSE  : {rmse:.2f} kWh")
    print(f"  R²    : {r2:.4f}  ({r2*100:.1f}%)")
    print(f"  MAPE  : {mape:.2f}%")
    print(f"  Accuracy : {100-mape:.1f}%")

    metrics = {"MAE": round(mae,3), "RMSE": round(rmse,3), "R2": round(r2,4), "MAPE": round(mape,3)}
    with open(f"{OUT}/model_metrics.json","w") as f: json.dump(metrics, f, indent=2)

    coef_df = pd.DataFrame({"feature": FEATURES, "coef": model.coef_})
    coef_df.to_csv(f"{OUT}/feature_coefficients.csv", index=False)

    df_test = pd.DataFrame({"actual": y_test.values, "predicted": y_pred},
                            index=df["datetime"].iloc[-len(y_test):])
    df_test.to_csv(f"{OUT}/test_predictions.csv")

    print("\n[3/5] Plotting demand overview...")
    plot_overview(df)

    print("[4/5] Plotting diagnostics & peak analysis...")
    plot_regression_diagnostics(y_test.values, y_pred, coef_df)
    plot_actual_vs_predicted(df_test.reset_index())
    plot_peak_analysis(df)

    print("[5/5] Generating 7-day forecast...")
    fcast = plot_forecast(model, scaler, df)
    fcast.to_csv(f"{OUT}/7day_forecast.csv", index=False)

    print("\n" + "═"*60)
    print("  DONE — Files saved to:", OUT)
    print("═"*60)
    for f in sorted(os.listdir(OUT)): print(f"  ✓ {f}")
