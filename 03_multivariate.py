import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

t5 = pd.read_csv('analysis/data/table5_graph_descriptors.csv').set_index('Variant')
t6 = pd.read_csv('analysis/data/table6_molar_mass_descriptors.csv').set_index('Node').T  # variants as rows
t6.index.name='Variant'

conv200 = pd.read_csv('analysis/data/convergence_200ns.csv').set_index('Variant')
conv200.index = conv200.index.map({'WT':'Wildtype','Y100C':'Y100C','C124Y':'C124Y','S132F':'S132F','M133K':'M133K','G145R':'G145R'})

# 50ns summary (thesis-reported, verified)
md50 = pd.DataFrame({
 'Variant':['Wildtype','Y100C','C124Y','S132F','M133K','G145R'],
 'Mean_RMSD_50ns':[0.457,0.491,0.486,0.534,0.547,0.503],
 'Max_RMSD_50ns':[0.534,0.638,0.611,0.592,0.683,0.585]}).set_index('Variant')

# node-level summary features per variant from Table 6
node_feat = pd.DataFrame({
    'Node_mean_dev': t6.mean(axis=1),
    'Node_sd_dev': t6.std(axis=1),
    'Node_max_dev': t6.max(axis=1),
})

master = t5.join(node_feat).join(conv200[['Mean_RMSD','Max_RMSD']].rename(
    columns={'Mean_RMSD':'Mean_RMSD_200ns','Max_RMSD':'Max_RMSD_200ns'})).join(md50)
master.to_csv('analysis/data/master_feature_table.csv')
print(master)

# ---- Correlation heatmap: graph descriptors vs MD stability metrics ----
corr_vars = ['d1_density','d2_diameter','d3_avg_shortest_path','d4_avg_degree',
             'd5_max_betweenness','d6_avg_betweenness','d7_max_eigenvalue','d8_avg_eigenvalue',
             'Node_mean_dev','Node_sd_dev','Node_max_dev']
target_vars = ['Mean_RMSD_50ns','Max_RMSD_50ns','Mean_RMSD_200ns','Max_RMSD_200ns']

rho_mat = pd.DataFrame(index=corr_vars, columns=target_vars, dtype=float)
p_mat = pd.DataFrame(index=corr_vars, columns=target_vars, dtype=float)
for cv in corr_vars:
    for tv in target_vars:
        rho, p = stats.spearmanr(master[cv], master[tv])
        rho_mat.loc[cv,tv]=rho
        p_mat.loc[cv,tv]=p
rho_mat.to_csv('analysis/data/spearman_rho.csv')
p_mat.to_csv('analysis/data/spearman_p.csv')

fig, ax = plt.subplots(figsize=(6,7))
im = ax.imshow(rho_mat.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(target_vars))); ax.set_xticklabels(target_vars, rotation=30, ha='right')
ax.set_yticks(range(len(corr_vars))); ax.set_yticklabels(corr_vars)
for i in range(len(corr_vars)):
    for j in range(len(target_vars)):
        r = rho_mat.values[i,j]; p = p_mat.values[i,j]
        star = '*' if p<0.05 else ('~' if p<0.10 else '')
        ax.text(j,i,f'{r:.2f}{star}', ha='center', va='center', fontsize=8,
                color='white' if abs(r)>0.6 else 'black')
plt.colorbar(im, ax=ax, label="Spearman's rho (n=6)")
ax.set_title("Graph-theoretic/molar-mass descriptors vs.\nMD stability metrics (Spearman correlation)")
plt.tight_layout()
plt.savefig('analysis/figures/fig3_correlation_heatmap.png', dpi=300)
plt.close()

# ---- PCA on graph descriptors (Table 5) ----
X = StandardScaler().fit_transform(t5.values)
pca = PCA(n_components=3)
scores = pca.fit_transform(X)
evr = pca.explained_variance_ratio_
pca_df = pd.DataFrame(scores, index=t5.index, columns=['PC1','PC2','PC3'])
pca_df.to_csv('analysis/data/pca_scores_graph_descriptors.csv')

fig, ax = plt.subplots(figsize=(6.5,5.5))
colors_map = {'Wildtype':'black','Y100C':'#E69F00','C124Y':'#009E73','S132F':'#CC79A7','M133K':'#0072B2','G145R':'#D55E00'}
for v in pca_df.index:
    ax.scatter(pca_df.loc[v,'PC1'], pca_df.loc[v,'PC2'], s=120, color=colors_map[v], label=v, edgecolor='k', zorder=3)
    ax.annotate(v, (pca_df.loc[v,'PC1'], pca_df.loc[v,'PC2']), textcoords='offset points', xytext=(6,6), fontsize=9)
# loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
scale = 2.2
for i, feat in enumerate(t5.columns):
    ax.arrow(0,0, loadings[i,0]*scale, loadings[i,1]*scale, color='gray', alpha=0.6, width=0.005, head_width=0.08)
    ax.text(loadings[i,0]*scale*1.15, loadings[i,1]*scale*1.15, feat, fontsize=7, color='dimgray', ha='center')
ax.axhline(0,color='lightgray',lw=0.8); ax.axvline(0,color='lightgray',lw=0.8)
ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}% variance)')
ax.set_title('PCA biplot of weighted graph-theoretic descriptors (d1\u2013d8)')
plt.tight_layout()
plt.savefig('analysis/figures/fig4_pca_biplot.png', dpi=300)
plt.close()
print('PCA explained variance ratio:', evr)
print(pca_df)

# ---- Hierarchical clustering (formal redo) on Table 6 node-level descriptors, Manhattan distance ----
D = pdist(t6.values, metric='cityblock')
Z = linkage(D, method='average')
coph_corr, coph_dists = cophenet(Z, D)
print('Cophenetic correlation coefficient:', coph_corr)

fig, ax = plt.subplots(figsize=(7,5))
dendrogram(Z, labels=t6.index.tolist(), ax=ax, color_threshold=0.9*max(Z[:,2]))
ax.set_ylabel('Manhattan distance')
ax.set_title(f'Hierarchical clustering of HBsAg variants\n(node molar-mass deviation profile; cophenetic r = {coph_corr:.2f})')
plt.tight_layout()
plt.savefig('analysis/figures/fig5_dendrogram.png', dpi=300)
plt.close()

# Silhouette analysis for choosing k (k=2..4) using precomputed distance matrix
from sklearn.cluster import AgglomerativeClustering
Dsq = squareform(D)
sil_rows=[]
for k in [2,3,4]:
    model = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
    labels_k = model.fit_predict(Dsq)
    sil = silhouette_score(Dsq, labels_k, metric='precomputed')
    sil_rows.append({'k':k,'silhouette':sil, 'labels': dict(zip(t6.index, labels_k))})
    print(k, sil, dict(zip(t6.index, labels_k)))
pd.DataFrame([{'k':r['k'],'silhouette':r['silhouette']} for r in sil_rows]).to_csv('analysis/data/silhouette_scores.csv', index=False)
