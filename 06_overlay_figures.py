import pandas as pd, numpy as np, pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df200 = pd.read_csv('analysis/data/rmsd_200ns.csv')
with open('analysis/data/rmsd_50ns.pkl','rb') as f:
    d50 = pickle.load(f)

variants = ['WT','Y100C','C124Y','S132F','M133K','G145R']
colors = {'WT':'black','Y100C':'#E69F00','C124Y':'#009E73','S132F':'#CC79A7','M133K':'#0072B2','G145R':'#D55E00'}
disp = {'WT':'Wildtype'}

fig, axes = plt.subplots(1,2, figsize=(12,5))
for v in variants:
    axes[0].plot(d50[v]['Time_ns'], d50[v]['RMSD'], color=colors[v], lw=1.1, label=disp.get(v,v))
axes[0].set_xlabel('Time (ns)'); axes[0].set_ylabel('RMSD (nm)'); axes[0].set_title('50 ns simulation')
axes[0].legend(frameon=False, fontsize=8)
axes[0].spines[['top','right']].set_visible(False)

for v in variants:
    axes[1].plot(df200['Time_ns'], df200[v], color=colors[v], lw=0.7, label=disp.get(v,v), alpha=0.9)
axes[1].set_xlabel('Time (ns)'); axes[1].set_ylabel('RMSD (nm)'); axes[1].set_title('200 ns simulation')
axes[1].legend(frameon=False, fontsize=8)
axes[1].spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('analysis/figures/fig0_rmsd_overlay.png', dpi=300)
plt.close()
print('done')
