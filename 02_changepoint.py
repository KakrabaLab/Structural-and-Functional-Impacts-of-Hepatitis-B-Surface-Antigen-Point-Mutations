import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df200 = pd.read_csv('analysis/data/rmsd_200ns.csv')
variants = ['WT','Y100C','C124Y','S132F','M133K','G145R']
colors = {'WT':'black','Y100C':'#E69F00','C124Y':'#009E73','S132F':'#CC79A7','M133K':'#0072B2','G145R':'#D55E00'}
t = df200['Time_ns'].values

def sse_single_split(x):
    """Return best single split index (mean-shift, sum-of-squares cost) and cost reduction."""
    n = len(x)
    cs = np.cumsum(x)
    cs2 = np.cumsum(x**2)
    total_sse = cs2[-1] - cs[-1]**2/n
    best_gain, best_k = -1, None
    for k in range(50, n-50):
        s1, s2 = cs[k-1], cs[-1]-cs[k-1]
        ss1, ss2 = cs2[k-1], cs2[-1]-cs2[k-1]
        n1, n2 = k, n-k
        sse1 = ss1 - s1**2/n1
        sse2 = ss2 - s2**2/n2
        gain = total_sse - (sse1+sse2)
        if gain > best_gain:
            best_gain, best_k = gain, k
    return best_k, best_gain, total_sse

def binary_segmentation(x, t, max_segments=4, min_gain_frac=0.03):
    """Recursively split; stop when gain is small relative to total variance."""
    segments = [(0, len(x))]
    changepoints = []
    total_var = np.sum((x-x.mean())**2)
    while len(segments) < max_segments:
        best = None
        for (a,b) in segments:
            if b-a < 150: continue
            k, gain, _ = sse_single_split(x[a:b])
            if k is None: continue
            if best is None or gain > best[2]:
                best = (a,b,gain,k)
        if best is None: break
        a,b,gain,k = best
        if gain/total_var < min_gain_frac: break
        cp = a+k
        changepoints.append(cp)
        segments.remove((a,b))
        segments += [(a,cp),(cp,b)]
    return sorted(changepoints), sorted(segments)

results = {}
fig, axes = plt.subplots(3,2, figsize=(11,10), sharex=True)
for i,v in enumerate(variants):
    x = df200[v].values
    cps, segs = binary_segmentation(x, t, max_segments=4)
    results[v] = {'changepoints_ns': [t[c] for c in cps]}
    ax = axes.flat[i]
    ax.plot(t, x, color=colors[v], lw=0.5, alpha=0.85)
    seg_bounds = sorted(set([0]+cps+[len(x)]))
    for a,b in zip(seg_bounds[:-1], seg_bounds[1:]):
        seg_mean = x[a:b].mean()
        ax.hlines(seg_mean, t[a], t[b-1], color='k', lw=2, alpha=0.8)
    for c in cps:
        ax.axvline(t[c], color='red', ls='--', lw=1)
    ax.set_title(('Wildtype' if v=='WT' else v))
    ax.set_ylabel('RMSD (nm)')
    if i>=4: ax.set_xlabel('Time (ns)')
plt.tight_layout()
plt.savefig('analysis/figures/fig2_changepoint_segmentation.png', dpi=300)
plt.close()

rows=[]
for v in variants:
    cps = results[v]['changepoints_ns']
    rows.append({'Variant': v, 'N_changepoints': len(cps), 'Changepoints_ns': ', '.join(f'{c:.2f}' for c in cps)})
cp_df = pd.DataFrame(rows)
cp_df.to_csv('analysis/data/changepoints_200ns.csv', index=False)
print(cp_df)
