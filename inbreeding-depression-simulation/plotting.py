# plotting.py
# Unified plotting module containing all visualization functions

import matplotlib
# LaTeX-style fonts (must be set BEFORE creating any figure):
# Computer Modern serif text + Computer Modern math, matching LaTeX output.
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['cmr10', 'Computer Modern Roman', 'DejaVu Serif']
matplotlib.rcParams['axes.formatter.use_mathtext'] = True

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d


def _save_png_and_pdf(fig, png_path, dpi=300, tight=True):
    """
    Save a Matplotlib figure as both PNG and vector PDF.
    """
    base_path, _ = os.path.splitext(png_path)
    if tight:
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
        fig.savefig(f"{base_path}.pdf", format="pdf",
                    bbox_inches="tight", pad_inches=0.25)
    else:
        fig.savefig(png_path, dpi=dpi)
        fig.savefig(f"{base_path}.pdf", format="pdf")


# =============================================================================
#  Robust Simulation Plots
# =============================================================================

def create_plots(results, output_dir):
    """
    Generate all plots from Robust Simulation comparative statics results.
    """
    print(f"\nGenerating plots, saving to '{output_dir}'...")

    label_fontsize = 26
    tick_fontsize = 22
    colorbar_fontsize = 24
    surface_label_fontsize = 22
    surface_tick_fontsize = 18
    line_width = 2.8

    sweep_param_name = results["sweep_param_name"]
    sweep_param_values = results["sweep_param_values"]
    ai_values = results["ai_values"]
    potential_benefits = results["potential_benefits"]
    optimal_ai_proportions = results["optimal_ai_proportions"]
    avg_yields_mesh = results["avg_yields_mesh"]

    xlabel_map = {
        'h': r'Dominance Coefficient ($h$)',
        'Delta': r'Selection Coefficient ($\Delta$)',
        'pi': r'Disease Incidence ($\pi$)',
        'gamma': r'Disease Severity ($\gamma$)',
        'num_generations': 'Number of Generations',
        'initial_A_proportion': r'Initial Allele Proportion ($a_0$)'
    }
    Y_label = xlabel_map.get(sweep_param_name, sweep_param_name)
    compact_label_map = {
        'h': r'$h$',
        'Delta': r'$\Delta$',
        'pi': r'$\pi$',
        'gamma': r'$\gamma$',
        'num_generations': r'$T$',
        'initial_A_proportion': r'$p_0$'
    }
    compact_y_label = compact_label_map.get(sweep_param_name, sweep_param_name)

    # Plot 1: Potential Benefit (percentage scale, fixed y-axis 0%–15%)
    import matplotlib.ticker as mticker
    plt.figure(figsize=(9.5, 7))
    plt.plot(sweep_param_values, potential_benefits * 100, 'b-', linewidth=line_width)
    plt.xlabel(compact_y_label, fontsize=label_fontsize)
    plt.ylabel('Potential Benefit (%)', fontsize=label_fontsize)
    plt.ylim(0, 15)
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    plt.grid(True)
    plt.tight_layout()
    _save_png_and_pdf(
        plt.gcf(),
        os.path.join(output_dir, f"benefit_vs_{sweep_param_name}.png")
    )

    # Plot 2: Optimal AI Proportion
    plt.figure(figsize=(9.5, 7))
    plt.plot(sweep_param_values, optimal_ai_proportions, 'b-', linewidth=line_width)
    plt.xlabel(compact_y_label, fontsize=label_fontsize)
    plt.ylabel(r'Optimal $a$', fontsize=label_fontsize)
    plt.ylim(0, 1)
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    plt.grid(True)
    plt.tight_layout()
    _save_png_and_pdf(
        plt.gcf(),
        os.path.join(output_dir, f"optimal_a_vs_{sweep_param_name}.png")
    )

    # Plot 3: 3D Surface
    X_MESH, Y_MESH = np.meshgrid(ai_values, sweep_param_values)
    surface_contour_figsize = (11.5, 8.2)

    fig = plt.figure(figsize=surface_contour_figsize)
    ax = fig.add_axes([0.06, 0.07, 0.78, 0.86], projection='3d')
    surf = ax.plot_surface(X_MESH, Y_MESH, avg_yields_mesh, cmap='viridis', edgecolor='none')
    ax.view_init(elev=24, azim=-58)
    try:
        ax.set_box_aspect((1.25, 1.0, 0.72), zoom=1.06)
    except TypeError:
        ax.set_box_aspect((1.25, 1.0, 0.72))
    ax.set_xlabel(r'$a$', fontsize=surface_label_fontsize, labelpad=8)
    ax.set_ylabel(compact_y_label, fontsize=surface_label_fontsize, labelpad=12)
    ax.set_zlabel('Average Yield', fontsize=surface_label_fontsize, labelpad=12)
    ax.tick_params(axis='both', labelsize=surface_tick_fontsize, pad=3)
    ax.tick_params(axis='z', labelsize=surface_tick_fontsize, pad=5)
    _save_png_and_pdf(
        fig,
        os.path.join(output_dir, f"yield_surface_vs_{sweep_param_name}.png"),
        tight=False
    )

    # Plot 4: Contour
    plt.figure(figsize=surface_contour_figsize)
    contour = plt.contourf(X_MESH, Y_MESH, avg_yields_mesh, levels=20, cmap='viridis')
    cbar = plt.colorbar(contour)
    cbar.set_label('Average Yield', fontsize=colorbar_fontsize, labelpad=16)
    cbar.ax.tick_params(labelsize=colorbar_fontsize)
    plt.xlabel(r'$a$', fontsize=label_fontsize)
    plt.ylabel(compact_y_label, fontsize=label_fontsize)
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    plt.tight_layout()
    _save_png_and_pdf(
        plt.gcf(),
        os.path.join(output_dir, f"yield_contour_vs_{sweep_param_name}.png")
    )

    print("All plots generated and saved as PNG and vector PDF files.")
    plt.show()


def create_band_plots(results, output_dir):
    """
    Generate publication-quality plots with confidence bands from replicated
    robust simulation results.
    """
    print(f"\nGenerating confidence-band plots, saving to '{output_dir}'...")

    sweep_param_name = results["sweep_param_name"]
    sweep_param_values = results["sweep_param_values"]
    ai_values = results["ai_values"]
    n_rep = results["n_replications"]

    xlabel_map = {
        'h': r'Dominance Coefficient ($h$)',
        'Delta': r'Selection Coefficient ($\Delta$)',
        'pi': r'Disease Incidence ($\pi$)',
        'gamma': r'Disease Severity ($\gamma$)',
        'num_generations': 'Number of Generations',
        'initial_A_proportion': r'Initial Allele Proportion ($a_0$)'
    }
    x_label = xlabel_map.get(sweep_param_name, sweep_param_name)

    # --- Plot 1: Optimal Policy Function with confidence band ---
    fig, ax = plt.subplots(figsize=(9, 6.5))

    mean_a = results["mean_optimal_a"]
    median_a = results["median_optimal_a"]
    ci_lo_a = results["ci_lower_optimal_a"]
    ci_hi_a = results["ci_upper_optimal_a"]

    ax.fill_between(sweep_param_values, ci_lo_a, ci_hi_a,
                     alpha=0.25, color='#2171B5', label='95% CI')
    ax.plot(sweep_param_values, mean_a, '-', color='#08519C',
            linewidth=2.2, label='Mean')
    ax.plot(sweep_param_values, median_a, '--', color='#6BAED6',
            linewidth=1.5, label='Median')

    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(r'Optimal Control Variabl ($a^*$)', fontsize=18)
    ax.set_ylim([0, 1])
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(fontsize=15, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_png_and_pdf(
        fig,
        os.path.join(output_dir, f"band_optimal_a_vs_{sweep_param_name}.png")
    )

    # --- Plot 2: Potential Benefit with confidence band (percentage) ---
    import matplotlib.ticker as mticker

    fig2, ax2 = plt.subplots(figsize=(9, 6.5))

    mean_b = results["mean_potential_benefits"] * 100
    ci_lo_b = results["ci_lower_benefits"] * 100
    ci_hi_b = results["ci_upper_benefits"] * 100

    ax2.fill_between(sweep_param_values, ci_lo_b, ci_hi_b,
                      alpha=0.25, color='#CB181D', label='95% CI')
    ax2.plot(sweep_param_values, mean_b, '-', color='#A50F15',
             linewidth=2.2, label='Mean')

    ax2.set_xlabel(x_label, fontsize=18)
    ax2.set_ylabel('Potential Benefit (%)', fontsize=18)
    ax2.set_ylim(0, 15)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax2.tick_params(axis='both', labelsize=15)
    ax2.legend(fontsize=15, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    _save_png_and_pdf(
        fig2,
        os.path.join(output_dir, f"band_benefit_vs_{sweep_param_name}.png")
    )

    # --- Plot 3: Spaghetti plot (all replications + mean) ---
    fig3, ax3 = plt.subplots(figsize=(9, 6.5))

    all_a = results["all_optimal_a"]
    for r in range(min(n_rep, 100)):  # cap at 100 traces for readability
        ax3.plot(sweep_param_values, all_a[r, :],
                 '-', color='#9ECAE1', alpha=0.15, linewidth=0.7)

    ax3.plot(sweep_param_values, mean_a, '-', color='#08519C',
             linewidth=2.5, label='Mean', zorder=10)
    ax3.fill_between(sweep_param_values, ci_lo_a, ci_hi_a,
                      alpha=0.2, color='#2171B5', label='95% CI', zorder=5)

    ax3.set_xlabel(x_label, fontsize=18)
    ax3.set_ylabel(r'Optimal Control Variable ($a^*$)', fontsize=18)
    ax3.set_ylim([0, 1])
    ax3.tick_params(axis='both', labelsize=15)
    ax3.legend(fontsize=15, loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    fig3.tight_layout()
    _save_png_and_pdf(
        fig3,
        os.path.join(output_dir, f"spaghetti_optimal_a_vs_{sweep_param_name}.png")
    )

    print("All confidence-band plots generated as PNG and vector PDF files.")
    plt.show()


# =============================================================================
#  Myopic Simulation Plots
# =============================================================================

def plot_p_evolution_comparison(results_list, initial_p_list, static_params, output_dir):
    """
    Plot comparison of dynamic evolution paths under different initial p values.
    """
    print(f"\nGenerating multi-initial-value evolution comparison, saving to '{output_dir}'...")

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(results_list)))

    if not results_list:
        print("Warning: No simulation results provided for plotting.")
        return

    total_generations = 0
    for i, results in enumerate(results_list):
        total_generations = results['total_generations']
        initial_p = initial_p_list[i]

        allele_p_series = results['allele_p_series']
        optimal_a_series = results['optimal_a_series']

        generations_axis = np.arange(1, total_generations + 1)

        plt.plot(generations_axis, allele_p_series, color=colors[i], marker='o',
                 markersize=4, linestyle='-', linewidth=2,
                 label=rf'Initial $p$ = {initial_p:.1f}')

        plt.plot(generations_axis, optimal_a_series, color=colors[i], marker='x',
                 markersize=5, linestyle='--', linewidth=2,
                 label=rf'$a$ (Initial $p$={initial_p:.1f})')

    h = static_params.get('h', 'N/A')
    Delta = static_params.get('Delta', 'N/A')

    # plt.title(f'Dynamics of p and a under Myopic Regulation\n'
    #           f'(h={h}, Δ={Delta})', fontsize=22)
    plt.xlabel('Generation', fontsize=18)
    plt.ylabel(r'$p$ or $a$', fontsize=18)
    plt.ylim(0, 1)
    plt.xlim(left=0.5, right=total_generations + 0.5)
    plt.legend(title="Initial Conditions", fontsize=14, title_fontsize=15)
    plt.tick_params(axis='both', labelsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)

    if total_generations < 25:
        plt.xticks(np.arange(1, total_generations + 1, step=1))
    else:
        plt.xticks(np.arange(0, total_generations + 1,
                              step=max(1, total_generations // 10)))

    plt.tight_layout()
    filepath = os.path.join(output_dir, "dynamic_p_evolution_comparison.png")
    _save_png_and_pdf(plt.gcf(), filepath)
    print(f"Evolution comparison plot saved to: {filepath} and matching PDF")
    plt.show()


def plot_policy_function(p_values, optimal_policy, model_params, output_dir):
    """
    Plot the optimal policy function a*(p).
    """
    plt.figure(figsize=(7, 7))
    plt.plot(p_values, optimal_policy, 'b-', linewidth=2, label=r'Optimal Policy $a^*(p)$')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label=r'$a = p$ (no intervention)')
    # plt.title(
    #     f'Myopic Optimal Policy Function a*(p)\n'
    #     f'(h={model_params["h"]}, Δ={model_params["Delta"]})',
    #     fontsize=20
    # )
    plt.xlabel(r"$p$ (Frequency of Allele A)", fontsize=18)
    plt.ylabel(r"$a^*$ (Optimal Policy)", fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tick_params(axis='both', labelsize=15)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(fontsize=15)
    plt.tight_layout()
    _save_png_and_pdf(
        plt.gcf(),
        os.path.join(output_dir, "policy_function_a_vs_p.png")
    )
    plt.show()


def plot_yield_curves(p_values_to_show, ai_values, static_params, results_dir):
    """
    Plot "expected yield vs. policy a" curves for selected p values,
    marking the optimal point on each curve.
    """
    from core import compute_expected_yield

    print(f"\n--- Generating yield curve explanation for p = {p_values_to_show} ---")
    plt.figure(figsize=(7, 7))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(p_values_to_show)))

    Delta = static_params['Delta']
    h = static_params['h']
    C1 = static_params.get('C1', 1.0)
    C2 = static_params.get('C2', 1.0)
    n_exp = static_params.get('n', 2)
    m_exp = static_params.get('m', 2)

    for i, p_val in enumerate(p_values_to_show):
        expected_yields = []
        for a_candidate in ai_values:
            yield_val = compute_expected_yield(
                p_val, a_candidate, Delta, h, C1, C2, n_exp, m_exp
            )
            expected_yields.append(yield_val)

        max_yield_idx = np.argmax(expected_yields)
        optimal_a = ai_values[max_yield_idx]
        max_yield = expected_yields[max_yield_idx]

        plt.plot(ai_values, expected_yields, color=colors[i],
                 label=rf'When $p$ = {p_val:.1f}')
        plt.plot(optimal_a, max_yield, 's', color='red', markersize=6,
                 markerfacecolor='none', markeredgecolor='red',
                 label=rf'Optimal Choice $a^*({p_val:.2f}) \approx {optimal_a:.2f}$')

    # plt.title('Maximizing Expected Yield of the Current Generation', fontsize=22)
    plt.xlabel(r'Control Variable: $a$ (Allele G in Controlled Breeding)', fontsize=18)
    plt.ylabel('Expected Yield of the Current Generation', fontsize=18)
    plt.tick_params(axis='both', labelsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()

    output_filepath = os.path.join(results_dir, "yield_curve_explanation.png")
    _save_png_and_pdf(plt.gcf(), output_filepath)
    print(f"Yield curve explanation plot saved to: {output_filepath} and matching PDF")
    plt.show()


# =============================================================================
#  DP Simulation Plots
# =============================================================================

def plot_dp_results(optimizer, p0_list=[0.2, 0.5, 0.8], output_dir=None):
    """
    Plot DP model four-panel results figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    # fig.suptitle(
    #     f'Parameters: h={optimizer.h}, Δ={optimizer.Delta}, '
    #     f'β={optimizer.beta}, cost_type={optimizer.cost_type}',
    #     fontsize=22, fontweight='bold'
    # )

    # 1. Value Function
    axes[0, 0].plot(optimizer.p_grid, optimizer.V, 'b-', linewidth=2)
    axes[0, 0].set_xlabel(r'$p$', fontsize=25)
    axes[0, 0].set_ylabel(r'$V(p)$', fontsize=25)
    axes[0, 0].set_ylim(0, 25)
    axes[0, 0].set_title('Value Function', fontsize=25)
    axes[0, 0].tick_params(axis='both', labelsize=20)
    axes[0, 0].grid(True)

    # 2. Optimal Policy
    axes[0, 1].plot(optimizer.p_grid, optimizer.policy, 'r-', linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label=r'$y=x$')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xlabel(r'$p$', fontsize=25)
    axes[0, 1].set_ylabel(r'$a$', fontsize=25)
    axes[0, 1].set_title('Optimal Policy Function', fontsize=25)
    axes[0, 1].legend(fontsize=25)
    axes[0, 1].tick_params(axis='both', labelsize=20)
    axes[0, 1].grid(True)

    # 3. Trajectory of p
    colors = ['blue', 'green', 'red']
    for i, p0 in enumerate(p0_list):
        p_traj, a_traj = optimizer.simulate_trajectory(p0, T=10)
        axes[1, 0].plot(p_traj, color=colors[i], linewidth=2,
                        label=rf'Initial $p={p0}$', marker='o', markersize=3)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel('Generation', fontsize=25)
    axes[1, 0].set_ylabel(r'$p_t$', fontsize=25)
    axes[1, 0].set_title(r'Dynamic Trajectory of $p$', fontsize=25)
    axes[1, 0].legend(fontsize=20)
    axes[1, 0].tick_params(axis='both', labelsize=20)
    axes[1, 0].grid(True)

    # 4. Trajectory of a
    for i, p0 in enumerate(p0_list):
        p_traj, a_traj = optimizer.simulate_trajectory(p0, T=10)
        axes[1, 1].plot(a_traj, color=colors[i], linewidth=2,
                        label=rf'Initial $p={p0}$', marker='s', markersize=3)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel('Generation', fontsize=20)
    axes[1, 1].set_ylabel(r'$a_t$', fontsize=25)
    axes[1, 1].set_title(r'Dynamic Trajectory of $a$', fontsize=25)
    axes[1, 1].legend(fontsize=20)
    axes[1, 1].tick_params(axis='both', labelsize=20)
    axes[1, 1].grid(True)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _save_png_and_pdf(plt.gcf(), os.path.join(output_dir, "dp_results.png"))

    plt.show()


def plot_bellman_objective(optimizer, p_values_to_plot=[0.2, 0.5, 0.8],
                           output_dir=None):
    """
    Visualize the Bellman equation RHS objective function.
    """
    a_grid = np.linspace(0, 1, 201)
    plt.figure(figsize=(10, 7))

    V_interpolator = interp1d(optimizer.p_grid, optimizer.V, kind='cubic',
                              fill_value="extrapolate")

    for p_val in p_values_to_plot:
        objective_values = []
        for a in a_grid:
            payoff = optimizer.immediate_payoff(a, p_val)
            next_p = optimizer.state_transition(a, p_val)
            future_value = V_interpolator(next_p)
            total_value = payoff + optimizer.beta * future_value
            objective_values.append(total_value)

        plt.plot(a_grid, objective_values, label=rf'$p = {p_val:.2f}$')

        max_val_idx = np.argmax(objective_values)
        optimal_a = a_grid[max_val_idx]
        max_val = objective_values[max_val_idx]
        plt.plot(optimal_a, max_val, 'o', markersize=8,
                 label=rf'$a^*({p_val:.2f}) \approx {optimal_a:.2f}$')

    plt.xlabel(r'$a$ (Control Variable)', fontsize=18)
    plt.ylabel(r"$R(a, p) + \beta V(p')$", fontsize=18)
    # plt.title('Bellman Equation Objective Function', fontsize=22)
    plt.tick_params(axis='both', labelsize=15)
    plt.grid(True)
    plt.legend(fontsize=14)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _save_png_and_pdf(
            plt.gcf(),
            os.path.join(output_dir, "bellman_objective.png")
        )

    plt.show()


# =============================================================================
#  Policy Comparison Plot
# =============================================================================

def plot_policy_comparison(p_grid_myopic, policy_myopic,
                           p_grid_dp, policy_dp,
                           shared_params, dp_params, output_dir):
    """
    Plot Myopic vs. Forward-looking policy function comparison.
    """
    plt.figure(figsize=(8, 8))

    plt.plot(p_grid_myopic, policy_myopic, 'b-', linewidth=2.5,
             label='Myopic Regulator')
    plt.plot(p_grid_dp, policy_dp, 'r--', linewidth=2.5,
             label='Forward-looking Regulator')
    plt.plot([0, 1], [0, 1], 'k:', linewidth=1.5, label=r'$a = p$')

    # title_str = (
    #     f'Comparison of Policy Functions\n'
    #     f'Shared: h={shared_params["h"]}, Δ={shared_params["Delta"]}\n'
    #     f'Dynamic: β={dp_params["beta"]}'
    # )
    # plt.title(title_str, fontsize=20)
    plt.xlabel(r"$p$ (state variable)", fontsize=20)
    plt.ylabel(r"$a$ (control variable)", fontsize=20)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tick_params(axis='both', labelsize=18)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=16)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    _save_png_and_pdf(plt.gcf(), os.path.join(output_dir, "policy_comparison.png"))
    print(f"Comparison plot saved to: {output_dir}/policy_comparison.png and matching PDF")
    plt.show()
