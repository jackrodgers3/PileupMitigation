import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
DATA_DIR = r"C:\Users\jackm\PycharmProjects\PileupMitigation\SSL_GNN_REVAMP\test_data\post_data/"

def plotting(data_dir):
    mass_pred = torch.load(
        data_dir + 'mass_jets_pred.pt')
    mass_truth = torch.load(
        data_dir + 'mass_jets_truth.pt')
    mass_puppi = torch.load(
        data_dir + 'mass_jets_puppi.pt')
    print(mass_puppi.shape)
    print(mass_pred.shape)
    print(mass_truth.shape)
    plt.hist(mass_truth, bins=60)
    plt.title("Truth mass histogram")
    plt.xlabel("Mass")
    plt.ylabel("Density")
    plt.xlim(right=300)
    plt.show()
    plt.close()


    with open(data_dir + 'perf_pred.pkl', 'rb') as f:
        perf_pred = pickle.load(f)
    f.close()
    ssl_data = perf_pred['gated_boost']

    with open(data_dir + 'perf_PUPPI.pkl', 'rb') as f:
        puppi_data = pickle.load(f)
    f.close()

    with open(data_dir + 'perf_CHS.pkl', 'rb') as f:
        chs_data = pickle.load(f)
    f.close()

    with open(data_dir + 'perf_PUPPI_wcut.pkl', 'rb') as f:
        pf_data = pickle.load(f)
    f.close()

    N_BINS = 40
    filter_mass_top = 150.
    filter_mass_bottom = 20.
    ssl_masses = [ssl_data[i].mass_truth for i in range(len(ssl_data)) if filter_mass_bottom <= ssl_data[i].mass_truth <= filter_mass_top]
    puppi_masses = [puppi_data[i].mass_truth for i in range(len(puppi_data)) if filter_mass_bottom <= puppi_data[i].mass_truth <= filter_mass_top]
    chs_masses = [chs_data[i].mass_truth for i in range(len(chs_data)) if filter_mass_bottom <= chs_data[i].mass_truth <= filter_mass_top]
    pf_masses = [pf_data[i].mass_truth for i in range(len(pf_data)) if filter_mass_bottom <= pf_data[i].mass_truth <= filter_mass_top]

    ssl_mass_rel_diffs = [ssl_data[i].mass_diff for i in range(len(ssl_data)) if filter_mass_bottom <= ssl_data[i].mass_truth <= filter_mass_top]
    puppi_mass_rel_diffs = [puppi_data[i].mass_diff for i in range(len(puppi_data)) if filter_mass_bottom <= puppi_data[i].mass_truth <= filter_mass_top]
    chs_mass_rel_diffs = [chs_data[i].mass_diff for i in range(len(chs_data)) if filter_mass_bottom <= chs_data[i].mass_truth <= filter_mass_top]
    pf_mass_rel_diffs = [pf_data[i].mass_diff for i in range(len(pf_data)) if filter_mass_bottom <= pf_data[i].mass_truth <= filter_mass_top]

    ssl_masses_binned = binned_statistic(x=ssl_masses, values=ssl_masses, statistic='mean', bins=N_BINS)
    ssl_masses_reldiff_means_binned = binned_statistic(x=ssl_masses, values=ssl_mass_rel_diffs, statistic='mean',
                                                       bins=N_BINS)
    ssl_masses_reldiff_stds_binned = binned_statistic(x=ssl_masses, values=ssl_mass_rel_diffs, statistic='std',
                                                      bins=N_BINS)

    puppi_masses_binned = binned_statistic(x=puppi_masses, values=puppi_masses, statistic='mean', bins=N_BINS)
    puppi_masses_reldiff_means_binned = binned_statistic(x=puppi_masses, values=puppi_mass_rel_diffs, statistic='mean',
                                                       bins=N_BINS)
    puppi_masses_reldiff_stds_binned = binned_statistic(x=puppi_masses, values=puppi_mass_rel_diffs, statistic='std',
                                                      bins=N_BINS)

    chs_masses_binned = binned_statistic(x=chs_masses, values=chs_masses, statistic='mean', bins=N_BINS)
    chs_masses_reldiff_means_binned = binned_statistic(x=chs_masses, values=chs_mass_rel_diffs, statistic='mean',
                                                         bins=N_BINS)
    chs_masses_reldiff_stds_binned = binned_statistic(x=chs_masses, values=chs_mass_rel_diffs, statistic='std',
                                                        bins=N_BINS)

    fig = plt.figure(figsize=(10, 8))
    plt.plot(ssl_masses_binned.statistic, np.absolute(ssl_masses_reldiff_means_binned.statistic), 'go', label='SSL')
    plt.plot(puppi_masses_binned.statistic, np.absolute(puppi_masses_reldiff_means_binned.statistic), 'ro', label='PUPPI')
    plt.plot(chs_masses_binned.statistic, np.absolute(chs_masses_reldiff_means_binned.statistic), 'bo', label='CHS')
    plt.title('Mean mass vs. mean relative difference in mass')
    plt.xlabel(r'mean mass [GeV]')
    plt.ylabel(r'mean of $(m_{reco} - m_{truth})/m_{truth}$')
    plt.yscale('log')
    plt.legend()
    plt.savefig('picture1.png')
    plt.close()

    plt.plot(ssl_masses_binned.statistic, np.absolute(ssl_masses_reldiff_stds_binned.statistic), 'go', label='SSL')
    plt.plot(puppi_masses_binned.statistic, np.absolute(puppi_masses_reldiff_stds_binned.statistic), 'ro',
             label='PUPPI')
    plt.plot(chs_masses_binned.statistic, np.absolute(chs_masses_reldiff_stds_binned.statistic), 'bo', label='CHS')
    plt.title('Mean mass vs. std relative difference in mass')
    plt.xlabel(r'mean mass [GeV]')
    plt.ylabel(r'std of $(m_{reco} - m_{truth})/m_{truth}$')
    plt.yscale('log')
    plt.legend()
    plt.savefig('picture2.png')
    plt.close()

    ssldpuppi = [(ssl_masses_reldiff_stds_binned.statistic[i] / puppi_masses_reldiff_stds_binned.statistic[i]) for i in range(N_BINS)]
    plt.plot(ssl_masses_binned.statistic, ssldpuppi, 'ro')
    plt.title('SSL / PUPPI rel mass diff std ratio')
    plt.grid(True)
    plt.savefig('picture3.png')


if __name__ == '__main__':
    plotting(DATA_DIR)