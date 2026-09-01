import pandas as pd, numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from scipy import stats

master = pd.read_csv('analysis/data/master_feature_table.csv', index_col=0)
feat_cols = ['d1_density','d5_max_betweenness','d6_avg_betweenness','Node_mean_dev','Node_max_dev']
X = master[feat_cols].values
y = master['Mean_RMSD_200ns'].values

loo = LeaveOneOut()
preds_ridge, preds_rf = [], []
for train_idx, test_idx in loo.split(X):
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    ridge = Ridge(alpha=2.0).fit(Xtr_s, ytr)
    preds_ridge.append(ridge.predict(Xte_s)[0])
    rf = RandomForestRegressor(n_estimators=300, max_depth=2, random_state=0).fit(Xtr, ytr)
    preds_rf.append(rf.predict(Xte)[0])

preds_ridge = np.array(preds_ridge); preds_rf = np.array(preds_rf)
def q2(y, pred):
    ss_res = np.sum((y-pred)**2); ss_tot = np.sum((y-y.mean())**2)
    return 1 - ss_res/ss_tot

r_ridge, p_ridge = stats.pearsonr(y, preds_ridge)
r_rf, p_rf = stats.pearsonr(y, preds_rf)
print('Ridge LOO Q2:', q2(y,preds_ridge), 'r=',r_ridge,'p=',p_ridge)
print('RF LOO Q2:', q2(y,preds_rf), 'r=', r_rf, 'p=', p_rf)

# Full-data RF feature importance (descriptive; not for causal claims given n=6)
rf_full = RandomForestRegressor(n_estimators=500, max_depth=2, random_state=0).fit(X,y)
importances = pd.Series(rf_full.feature_importances_, index=feat_cols).sort_values(ascending=False)
print(importances)

out = pd.DataFrame({'Variant':master.index,'Observed_MeanRMSD200':y,'Ridge_LOO_pred':preds_ridge,'RF_LOO_pred':preds_rf})
out.to_csv('analysis/data/loo_regression_predictions.csv', index=False)
importances.to_csv('analysis/data/rf_feature_importance.csv')

with open('analysis/data/regression_summary.txt','w') as f:
    f.write(f"Ridge regression LOO-CV: Q2={q2(y,preds_ridge):.3f}, Pearson r={r_ridge:.3f}, p={p_ridge:.3f}\n")
    f.write(f"Random Forest LOO-CV: Q2={q2(y,preds_rf):.3f}, Pearson r={r_rf:.3f}, p={p_rf:.3f}\n")
    f.write("Feature importances (RF, full data):\n")
    f.write(importances.to_string())
