"""Plotting helpers used by the simulation loop to produce per-run figures."""

import matplotlib.pyplot as plt


def plot_total_allocations(services, allocation_history, number_days, save_path=None):
    """
    Description
    - Plot the total number of travelers on each service across all simulation
      days.

    Parameters
    - services: list of Service objects used in the simulation.
    - allocation_history: dict mapping service name to a list of per-day total
      allocations (length == number_days).
    - number_days: length of the simulation (number of days).
    - save_path: file path to save the figure; if None, the figure is shown.

    Output
    - Either displays the figure or writes a PNG to ``save_path``.
    """
    plt.figure(figsize=(8, 5))
    for service in services:
        plt.plot(
            range(number_days),
            allocation_history[service.name],
            label=service.name,
            linewidth=2,
        )
    plt.title("Evolution of Total Service Allocations")
    plt.xlabel("Day")
    plt.ylabel("Number of Travelers (total)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_per_type_allocations(services, allocation_by_type, travelers, number_days, save_path=None):
    """
    Description
    - Plot the per-class allocation timeline as one subplot per traveler class.

    Parameters
    - services: list of Service objects used in the simulation.
    - allocation_by_type: dict mapping service name to a list (length == K)
      of per-day allocation lists for each traveler class.
    - travelers: list of Travelers objects (one entry per class).
    - number_days: length of the simulation (number of days).
    - save_path: file path to save the figure; if None, the figure is shown.

    Output
    - Either displays the figure or writes a PNG to ``save_path``.
    """
    fig, axes = plt.subplots(len(travelers), 1, figsize=(8, 4 * len(travelers)), sharex=True)
    if len(travelers) == 1:
        axes = [axes]

    for t_idx, ax in enumerate(axes):
        for service in services:
            y_vals = [day_vals[t_idx] for day_vals in zip(*allocation_by_type[service.name])]
            ax.plot(range(number_days), y_vals, label=service.name, linewidth=2)
        ax.set_title(f"Traveler Type {t_idx + 1}")
        ax.set_ylabel("Number of Travelers")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[-1].set_xlabel("Day")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_gradient_evolution(gradient_history, tnc_names, save_path=None):
    """
    Description
    - Plot the per-operator gradient norm history across upper-level updates.

    Parameters
    - gradient_history: list of per-update dicts with keys
      ``update_idx``, ``grad_<tnc_name>_norm`` for every TNC operator, and
      ``grad_maas_norm``.
    - tnc_names: list of TNC operator names whose gradient norms should be
      drawn (e.g. ``["TNC1", "TNC2"]``).
    - save_path: file path to save the figure; if None, the figure is shown.

    Output
    - Either displays the figure or writes a PNG to ``save_path``.
    """
    if not gradient_history:
        return

    updates = [item["update_idx"] for item in gradient_history]
    plt.figure(figsize=(8, 5))
    for name in tnc_names:
        key = f"grad_{name}_norm"
        vals = [item[key] for item in gradient_history]
        plt.plot(updates, vals, label=f"||grad_{name}||", linewidth=2)

    grad_maas_norm = [item["grad_maas_norm"] for item in gradient_history]
    plt.plot(updates, grad_maas_norm, label="||grad_MaaS||", linewidth=2)

    plt.title("Upper-Level Gradient Norm Evolution")
    plt.xlabel("Upper-Level Update Index")
    plt.ylabel("Gradient Norm")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_param_delta_evolution(gradient_history, tnc_names, save_path=None):
    """
    Description
    - Plot the per-operator normalized parameter change across upper-level updates.

    Parameters
    - gradient_history: list of per-update dicts with keys
            ``update_idx``, ``param_delta_rel_<tnc_name>`` for every TNC operator,
            and ``param_delta_rel_MaaS``.
    - tnc_names: list of TNC operator names whose objective deltas should be
      drawn (e.g. ``["TNC1", "TNC2"]``).
    - save_path: file path to save the figure; if None, the figure is shown.

    Output
    - Either displays the figure or writes a PNG to ``save_path``.
    """
    if not gradient_history:
        return

    required_keys = [f"param_delta_rel_{name}" for name in tnc_names] + ["param_delta_rel_MaaS"]
    if not all(required_key in gradient_history[0] for required_key in required_keys):
        return

    updates = [item["update_idx"] for item in gradient_history]
    plt.figure(figsize=(8, 5))
    for name in tnc_names:
        key = f"param_delta_rel_{name}"
        vals = [item[key] for item in gradient_history]
        plt.plot(updates, vals, label=f"Δparam_rel_{name}", linewidth=2)

    maas_vals = [item["param_delta_rel_MaaS"] for item in gradient_history]
    plt.plot(updates, maas_vals, label="Δparam_rel_MaaS", linewidth=2)

    plt.title("Upper-Level Normalized Parameter Change per Update")
    plt.xlabel("Upper-Level Update Index")
    plt.ylabel("Relative Parameter Change")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
