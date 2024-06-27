import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import modules.profiles as prof
from limepy import limepy
import limepy_fit_surf_brightness as lp
import corner

class GCAnalyzer:
    def __init__(self, folder_path, base_filename='snp_w05_', num_bins=20, start=-0.1):
        self.folder_path = folder_path
        self.base_filename = base_filename
        self.num_bins = num_bins
        self.start = start
        self.df = None
        self.bin_edges = None
        self.sb_data = None
        self.profile = None
        self.samples = None
        self.loglike = None
        self.models = None
        self.Rh_Ltot = None
        self.rhl_true = None
        self.lv_tot_true = None
        self.sim_data = None
        self.load_sim_data()
        self.results_df = pd.DataFrame()

    def load_sim_data(self):
        sim_data_path = 'table_sim_w05_v2.dat'
        self.sim_data = pd.read_csv(sim_data_path, delim_whitespace=True)

    def load_data(self, file_number):
        file_path = os.path.join(self.folder_path, f"{self.base_filename}{file_number:04d}.dat")
        self.df = pd.read_csv(file_path, sep='\s+', header=None)
        self.df.columns = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'LV', 'M', 'B-V', 'V', 'binary_type', 'type-01', 'type-02']
        self.df[['x', 'y', 'z', 'vx', 'vy', 'vz', 'LV', 'M', 'B-V', 'V']] = self.df[['x', 'y', 'z', 'vx', 'vy', 'vz', 'LV', 'M', 'B-V', 'V']].apply(pd.to_numeric, errors='coerce')
        self.df['R'] = np.sqrt(self.df['x']**2 + self.df['y']**2)

    def calculate_true_values(self):
        self.df = self.df.sort_values('R')
        self.df['cumulative_LV'] = self.df['LV'].cumsum()
        self.lv_tot_true = self.df['LV'].sum()
        half_light = self.lv_tot_true / 2
        self.rhl_true = self.df.loc[self.df['cumulative_LV'] >= half_light, 'R'].iloc[0]

    def plot_cumulative_luminosity(self, file_number):
        plt.figure(figsize=(10, 6))
        plt.plot(self.df['R'], self.df['cumulative_LV'] / self.lv_tot_true)
        plt.axhline(y=0.5, color='r', linestyle='--')
        plt.axvline(x=self.rhl_true, color='g', linestyle='--', label='Computed Half-light Radius')
        
        sim_rhl = self.sim_data.loc[self.sim_data['name'] == f'snp_w05_{file_number:04d}', 'rhl'].values[0]
        plt.axvline(x=sim_rhl, color='b', linestyle='--', label='Simulation Half-light Radius')
        
        plt.xlabel('Radius')
        plt.ylabel('Normalized Cumulative Luminosity')
        plt.title(f'Cumulative Luminosity Profile (Simulation {file_number:04d})')
        plt.xscale('log')
        plt.text(self.rhl_true, 0.52, f'Computed Half-light radius: {self.rhl_true:.2f}', 
                 verticalalignment='bottom', horizontalalignment='left', color='g')
        plt.text(sim_rhl, 0.48, f'Simulation Half-light radius: {sim_rhl:.2f}', 
                 verticalalignment='top', horizontalalignment='left', color='b')
        plt.legend()
        plt.savefig(f'cumulative_luminosity_{file_number:04d}.png')
        plt.close()

    def create_radial_bins(self):
        self.bin_edges = np.logspace(self.start, np.log10(self.df['R'].max()), self.num_bins)
        self.df['radial_bin'] = pd.cut(self.df['R'], bins=self.bin_edges)

    def calculate_profiles(self):
        luminosity_bins = self.df.groupby('radial_bin')['LV'].sum()
        bin_areas = np.pi * (self.bin_edges[1:]**2 - self.bin_edges[:-1]**2)
        surface_brightness = luminosity_bins / bin_areas
        velocity_dispersion = self.df.groupby('radial_bin')['vz'].std()

        self.sb_data = pd.DataFrame({
            'radial_bin': luminosity_bins.index,
            'surface_brightness': surface_brightness,
            'velocity_dispersion': velocity_dispersion
        })
        self.sb_data['radius'] = self.sb_data['radial_bin'].apply(lambda x: x.mid)

    def plot_profiles(self, file_number):
        plt.figure(figsize=(8, 6))
        plt.plot(self.sb_data['radius'], self.sb_data['surface_brightness'], marker='o')
        plt.xlabel('Radius')
        plt.ylabel('Surface Brightness')
        plt.title(f'Surface Brightness vs. Radius (Simulation {file_number:04d})')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(f'surface_brightness_{file_number:04d}.png')
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(self.sb_data['radius'], self.sb_data['velocity_dispersion'], marker='o')
        plt.xlabel('Radius')
        plt.ylabel('Velocity Dispersion')
        plt.title(f'Velocity Dispersion vs. Radius (Simulation {file_number:04d})')
        plt.xscale('log')
        plt.grid(True)
        plt.savefig(f'velocity_dispersion_{file_number:04d}.png')
        plt.close()

    def calculate_lum_proj_average(self):
        self.profile = prof.LUM_PROJ_AVERAGE_FIX(self.df['x'], self.df['y'], self.df['z'], self.df['LV'], self.df['M'], self.bin_edges, n_samples=100)

    def fit_limepy_model(self, model_types=[1, 2], init_guess=np.array([5,4,3])):
        self.samples = {}
        self.loglike = {}
        self.models = {}
        self.Rh_Ltot = {}
        
        for model_type in model_types:
            samples, loglike, models, Rh_Ltot = lp.get_full_limpy_fit(
                self.profile[0,0], self.profile[2,0], np.amax(self.profile[2,1:], axis=0),
                model_type, init_guess, verbose=True, progress=True, n_walkers=1000
            )
            self.samples[model_type] = samples
            self.loglike[model_type] = loglike
            self.models[model_type] = models
            self.Rh_Ltot[model_type] = Rh_Ltot

    def plot_limepy_fit(self, file_number):
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red']
        labels = ['King Model (g=1)', 'Wilson Model (g=2)']
        
        for i, model_type in enumerate([1, 2]):
            num_models = len(self.models[model_type]) - 1  # Subtract 1 because the first element is the x-axis
            for j in range(num_models):
                if j == 0:  # Only a dfhjkdd label for the first line of each model
                    plt.plot(self.models[model_type][0], self.models[model_type][j+1], alpha=0.1, color=colors[i], label=labels[i])
                else:
                    plt.plot(self.models[model_type][0], self.models[model_type][j+1], alpha=0.1, color=colors[i])
        
        plt.errorbar(self.profile[0,0], self.profile[2,0], yerr=[self.profile[2,1], self.profile[2,2]], fmt='o', color='black', label='Data')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-3, 1e4)
        plt.xlabel('Radius')
        plt.ylabel('Surface Brightness')
        plt.title(f'Surface Brightness vs. Radius (LIMEPY Fits) (Simulation {file_number:02d})')
        plt.legend()
        plt.savefig(f'limepy_fit_{file_number:02d}.png')
        plt.close()

    def plot_rh_ltot_hist(self, file_number):
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        colors = ['blue', 'red']
        labels = ['King Model (g=1)', 'Wilson Model (g=2)']
        
        for i, model_type in enumerate([1, 2]):
            ax[0].hist(self.Rh_Ltot[model_type][:,0], bins=50, histtype='step', color=colors[i], label=labels[i])
            ax[1].hist(self.Rh_Ltot[model_type][:,1], bins=50, histtype='step', color=colors[i], label=labels[i])
        
        ax[0].axvline(self.rhl_true, color='green', linestyle='--', label='Computed Value')
        sim_rhl = self.sim_data.loc[self.sim_data['name'] == f'snp_w05_{file_number:04d}', 'rhl'].values[0]
        ax[0].axvline(sim_rhl, color='black', linestyle='--', label='Simulation Value')
        ax[0].set_xlabel('Half-light Radius')
        ax[0].set_ylabel('Frequency')
        ax[0].legend()

        ax[1].axvline(self.lv_tot_true, color='green', linestyle='--', label='Computed Value')
        sim_lv_tot = self.sim_data.loc[self.sim_data['name'] == f'snp_w05_{file_number:04d}', 'Lv_tot'].values[0]
        ax[1].axvline(sim_lv_tot, color='black', linestyle='--', label='Simulation Value')
        ax[1].set_xlabel('Total Luminosity')
        ax[1].set_ylabel('Frequency')
        ax[1].legend()

        plt.suptitle(f'Distribution of LIMEPY Model Parameters (Simulation {file_number:02d})')
        plt.savefig(f'rh_ltot_hist_{file_number:02d}.png')
        plt.close()

    def plot_corner(self, file_number):
        labels = ["W0", r'$r_hl$', r"$\log_{10}(L)$"]
        colors = ['blue', 'red']
        model_names = ['King Model (g=1)', 'Wilson Model (g=2)']

        # Create a single figure for both models
        fig = corner.corner(self.samples[1], labels=labels, color=colors[0], 
                            plot_density=False, plot_datapoints=True, 
                            data_kwargs={'alpha':0.5, 'ms':1.5})

        # Add the Wilson model to the same figure
        corner.corner(self.samples[2], fig=fig, labels=labels, color=colors[1], 
                    plot_density=False, plot_datapoints=True, 
                    data_kwargs={'alpha':0.5, 'ms':1.5})

        # Adjust the plot
        axes = np.array(fig.axes).reshape((3, 3))
        for i in range(3):
            for j in range(3):
                if i > j:
                    ax = axes[i, j]
                    ax.set_title('')
                    # Add legend only to the first subplot
                    if i == 2 and j == 0:
                        for k, (color, name) in enumerate(zip(colors, model_names)):
                            ax.scatter([], [], color=color, alpha=0.5, s=10, label=name)
                        ax.legend(loc='center', fontsize='x-small')
        plt.suptitle(f"LIMEPY Model Parameter Correlations (Simulation {file_number:04d})")
        plt.savefig(f'corner_plot_combined_{file_number:04d}.png')
        plt.close(fig)
    
    def store_results(self, file_number):
        results = {
            'simulation': file_number,
            'computed_rhl': self.rhl_true,
            'computed_Lvtot': self.lv_tot_true,
            'sim_rhl': self.sim_data.loc[self.sim_data['name'] == f'snp_w05_{file_number:04d}', 'rhl'].values[0],
            'sim_Lvtot': self.sim_data.loc[self.sim_data['name'] == f'snp_w05_{file_number:04d}', 'Lv_tot'].values[0]
        }

        for model_type, model_name in zip([1, 2], ['King', 'Wilson']):
            fit_results = self.samples[model_type].mean(axis=0)
            fit_errors_upper = np.percentile(self.samples[model_type], 84, axis=0) - fit_results
            fit_errors_lower = fit_results - np.percentile(self.samples[model_type], 16, axis=0)

            results.update({
                f'{model_name}_W0': fit_results[0],
                f'{model_name}_W0_err_upper': fit_errors_upper[0],
                f'{model_name}_W0_err_lower': fit_errors_lower[0],
                f'{model_name}_rhl_3d': fit_results[1],
                f'{model_name}_rhl_3d_err_upper': fit_errors_upper[1],
                f'{model_name}_rhl_3d_err_lower': fit_errors_lower[1],
                f'{model_name}_log_Ltot': fit_results[2],
                f'{model_name}_log_Ltot_err_upper': fit_errors_upper[2],
                f'{model_name}_log_Ltot_err_lower': fit_errors_lower[2],
                f'{model_name}_Rh_2d': self.Rh_Ltot[model_type][:, 0].mean(),
                f'{model_name}_Rh_2d_err_upper': np.percentile(self.Rh_Ltot[model_type][:, 0], 84) - self.Rh_Ltot[model_type][:, 0].mean(),
                f'{model_name}_Rh_2d_err_lower': self.Rh_Ltot[model_type][:, 0].mean() - np.percentile(self.Rh_Ltot[model_type][:, 0], 16),
                f'{model_name}_Ltot': 10**fit_results[2],
                f'{model_name}_Ltot_err_upper': 10**(fit_results[2] + fit_errors_upper[2]) - 10**fit_results[2],
                f'{model_name}_Ltot_err_lower': 10**fit_results[2] - 10**(fit_results[2] - fit_errors_lower[2])
            })

        new_row = pd.DataFrame([results])
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)

    def run_analysis(self, file_number):
        self.load_data(file_number)
        self.calculate_true_values()
        self.plot_cumulative_luminosity(file_number)
        self.create_radial_bins()
        self.calculate_profiles()
        self.plot_profiles(file_number)
        self.calculate_lum_proj_average()
        self.fit_limepy_model([1, 2])  # Run both King (g=1) and Wilson (g=2) models
        self.plot_limepy_fit(file_number)
        self.plot_rh_ltot_hist(file_number)
        self.plot_corner(file_number)
        self.store_results(file_number)
    
    def save_results_to_csv(self, filename='gc_analysis_results.csv'):
        self.results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

def analyze_all_simulations(folder_path, start_num=0, end_num=399):
    analyzer = GCAnalyzer(folder_path)
    for i in range(start_num, end_num + 1):
        print(f"Analyzing simulation {i:04d}")
        analyzer.run_analysis(i)
    analyzer.save_results_to_csv()  # Add this line

if __name__ == "__main__":
    folder_path = ''
    analyze_all_simulations(folder_path)
