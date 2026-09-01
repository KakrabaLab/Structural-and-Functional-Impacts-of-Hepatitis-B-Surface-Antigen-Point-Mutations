import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})

df200 = pd.read_csv('analysis/data/rmsd_200ns.csv')
variants = ['WT','Y100C','C124Y','S132F','M133K','G145R']
colors = {'WT':'black','Y100C':'#E69F00','C124Y':'#009E73','S132F':'#CC79A7','M133K':'#0072B2','G145R':'#D55E00'}

def acf(x, nlags):
    x = x - x.mean()
    n = len(x)
    result = np.correlate(x, x, mode='full')[n-1:]
    result /= result[0]
    return result[:nlags]

def integrated_autocorr_time(x, c=5):
    a = acf(x, min(2000, len(x)//2))
    tau = 1.0
    for M in range(1, len(a)):
        tau = 1 + 2*np.sum(a[1:M+1])
        if M >= c*tau:
            return max(tau,1.0)
    return max(tau,1.0)

summary_rows = []
for v in variants:
    x = df200[v].values
    tau = integrated_autocorr_time(x)
    n_eff = len(x)/tau
    se_corrected = x.std(ddof=1)/np.sqrt(n_eff)
    summary_rows.append({
        'Variant': v, 'N_frames':len(x), 'Mean_RMSD':x.mean(), 'SD':x.std(ddof=1),
        'Max_RMSD':x.max(), 'Tau_int_frames':tau, 'N_eff':n_eff, 'SE_corrected':se_corrected
    })
conv200 = pd.DataFrame(summary_rows)
conv200.to_csv('analysis/data/convergence_200ns.csv', index=False)
print(conv200)

fig, ax = plt.subplots(figsize=(7,5))
for v in variants:
    x = df200[v].values
    running_mean = np.cumsum(x)/np.arange(1,len(x)+1)
    ax.plot(df200['Time_ns'], running_mean, label=('Wildtype' if v=='WT' else v), color=colors[v], lw=1.4)
ax.set_xlabel('Simulation time (ns)')
ax.set_ylabel('Cumulative mean RMSD (nm)')
ax.set_title('Convergence of mean RMSD over the 200 ns trajectories')
ax.legend(frameon=False, ncol=2, fontsize=8)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('analysis/figures/fig1_convergence_running_mean.png', dpi=300)
plt.close()

wt = df200['WT'].values
tau_wt = integrated_autocorr_time(wt)
neff_wt = len(wt)/tau_wt
mean_wt, sd_wt = wt.mean(), wt.std(ddof=1)

rows=[]
for v in variants:
    if v=='WT': continue
    x = df200[v].values
    tau_x = integrated_autocorr_time(x)
    neff_x = len(x)/tau_x
    mean_x, sd_x = x.mean(), x.std(ddof=1)
    se_diff = np.sqrt(sd_wt**2/neff_wt + sd_x**2/neff_x)
    t_stat = (mean_x - mean_wt)/se_diff
    df_ws = (sd_wt**2/neff_wt + sd_x**2/neff_x)**2 / (
        (sd_wt**2/neff_wt)**2/(neff_wt-1) + (sd_x**2/neff_x)**2/(neff_x-1))
    p_val = 2*(1-stats.t.cdf(abs(t_stat), df_ws))
    cohend = (mean_x-mean_wt)/np.sqrt((sd_wt**2+sd_x**2)/2)
    ks_stat, ks_p = stats.ks_2samp(wt, x)
    rows.append({'Variant':v,'Mean_diff_vs_WT':mean_x-mean_wt,'t_stat_eff':t_stat,'df_eff':df_ws,
                 'p_value_eff':p_val, 'Cohens_d':cohend, 'KS_stat':ks_stat,'KS_p':ks_p,
                 'N_eff_variant':neff_x})
stat_tests = pd.DataFrame(rows)
stat_tests.to_csv('analysis/data/hypothesis_tests_200ns.csv', index=False)
print(stat_tests)
