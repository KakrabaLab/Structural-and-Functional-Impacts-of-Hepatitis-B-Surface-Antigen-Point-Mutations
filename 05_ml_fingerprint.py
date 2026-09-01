import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

df200 = pd.read_csv('analysis/data/rmsd_200ns.csv')
variants = ['WT','Y100C','C124Y','S132F','M133K','G145R']

WIN = 125  # 125 frames * 0.04 ns = 5 ns windows
STEP = 25  # 1 ns stride

def featurize(x, t):
    feats, times = [], []
    for start in range(0, len(x)-WIN, STEP):
        seg = x[start:start+WIN]
        tseg = t[start:start+WIN]
        slope = np.polyfit(tseg, seg, 1)[0]
        f = {
            'mean': seg.mean(), 'std': seg.std(), 'min': seg.min(), 'max': seg.max(),
            'range': seg.max()-seg.min(), 'slope': slope,
            'skew': pd.Series(seg).skew(), 'kurt': pd.Series(seg).kurt(),
            'autocorr_lag1': pd.Series(seg).autocorr(lag=1) if len(seg)>2 else 0.0,
        }
        feats.append(f); times.append(tseg[0])
    return pd.DataFrame(feats), np.array(times)

all_rows = []
for v in variants:
    x = df200[v].values; t = df200['Time_ns'].values
    fdf, times = featurize(x, t)
    fdf['Variant'] = v
    fdf['window_start_ns'] = times
    all_rows.append(fdf)
full = pd.concat(all_rows, ignore_index=True)
full.to_csv('analysis/data/ml_window_features.csv', index=False)
print(full.shape)
print(full['Variant'].value_counts())

# Chronological (temporal) train/test split per variant: first 70% train, last 30% test
feat_cols = ['mean','std','min','max','range','slope','skew','kurt','autocorr_lag1']
train_mask = full.groupby('Variant')['window_start_ns'].transform(lambda s: s <= s.quantile(0.7))
train, test = full[train_mask], full[~train_mask]
Xtr, ytr = train[feat_cols].values, train['Variant'].values
Xte, yte = test[feat_cols].values, test['Variant'].values

clf = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=0, class_weight='balanced')
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)
acc = accuracy_score(yte, pred)
print('Temporal holdout accuracy:', acc)
report = classification_report(yte, pred, output_dict=True)
report_df = pd.DataFrame(report).T
report_df.to_csv('analysis/data/ml_classification_report.csv')
print(report_df)

cm = confusion_matrix(yte, pred, labels=variants)
cm_df = pd.DataFrame(cm, index=variants, columns=variants)
cm_df.to_csv('analysis/data/ml_confusion_matrix.csv')

fig, ax = plt.subplots(figsize=(6,5.5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(variants))); ax.set_xticklabels(['WT' if v=='WT' else v for v in variants], rotation=45, ha='right')
ax.set_yticks(range(len(variants))); ax.set_yticklabels(['WT' if v=='WT' else v for v in variants])
for i in range(len(variants)):
    for j in range(len(variants)):
        ax.text(j,i,cm[i,j], ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black', fontsize=9)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Random-forest classification of variant identity\nfrom 5-ns RMSD dynamics windows (temporal hold-out, acc={acc:.2f})')
plt.tight_layout()
plt.savefig('analysis/figures/fig6_ml_confusion_matrix.png', dpi=300)
plt.close()

importances = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
importances.to_csv('analysis/data/ml_feature_importance.csv')
print(importances)

fig, ax = plt.subplots(figsize=(6,4))
importances.sort_values().plot(kind='barh', ax=ax, color='#0072B2')
ax.set_xlabel('Random-forest feature importance (Gini)')
ax.set_title('Window-level features discriminating\nWT and HBsAg mutant RMSD dynamics')
plt.tight_layout()
plt.savefig('analysis/figures/fig7_ml_feature_importance.png', dpi=300)
plt.close()

# Baseline: shuffled-label control accuracy (permutation baseline)
rng = np.random.default_rng(0)
perm_accs = []
for i in range(200):
    ytr_perm = rng.permutation(ytr)
    clf_p = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0)
    clf_p.fit(Xtr, ytr_perm)
    perm_accs.append(accuracy_score(yte, clf_p.predict(Xte)))
perm_accs = np.array(perm_accs)
print('Permutation baseline accuracy: mean=', perm_accs.mean(), 'max=', perm_accs.max())
with open('analysis/data/ml_permutation_baseline.txt','w') as f:
    f.write(f"Permutation-label baseline (200 shuffles): mean acc={perm_accs.mean():.3f}, sd={perm_accs.std():.3f}, max={perm_accs.max():.3f}\n")
    f.write(f"Observed model accuracy: {acc:.3f}\n")
    f.write(f"Empirical p-value (observed >= permutation): {(np.sum(perm_accs>=acc)+1)/(len(perm_accs)+1):.4f}\n")
