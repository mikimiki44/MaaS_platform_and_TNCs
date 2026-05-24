from entities import (
    Service,
    TNC,
    MT,
    MaaS,
    Travelers,
    distribute_travelers,
    compute_utilities,
    compute_choice_probabilities,
    project_tnc_params,
    project_maas_params,
    store_allocations,
)
import autograd.numpy as np
from autograd import grad
import json
import os
from typing import Dict, List

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from plotting import (
    plot_total_allocations,
    plot_per_type_allocations,
    plot_gradient_evolution,
    plot_param_delta_evolution,
)
from debug_utils import (
    build_debug_snapshot,
    store_debug_snapshots,
    compute_operator_financials,
)


def run_simulation(
    tnc_capacities: Dict[str, float] | List[float],
    output_dir: str,
    number_days: int,
    debug_enabled: bool = False,
    enable_gradient_checks: bool = True,
    logit_scale_mu: float = 1,
):
    """
    Description
    - Run the full bi-level simulation for ``number_days`` days with the supplied
      TNC capacities, then write results, gradient history, and plots into
      ``output_dir``.

    Parameters
    - tnc_capacities: either a dict ``{tnc_name: capacity}`` or a positional
      list of capacities (auto-named ``TNC1, TNC2, ...``).
    - output_dir: directory to write outputs into (created if missing).
    - number_days: simulation horizon.
    - debug_enabled: when True, store the pre-update debug snapshot JSON.
    - enable_gradient_checks: when True, cross-check analytical gradients
      against autograd at every upper-level update.
    - logit_scale_mu: logit scale parameter (passed to travelers and services).

    Output
    - Writes ``final_results.json``, ``gradient_history.json``,
      ``total_allocation.png``, ``allocation_by_type.png``,
      ``gradient_evolution.png``, and (if ``debug_enabled``) the debug snapshot.
    """
    travelers = [
        Travelers(number_traveler=20, trip_length=30, value_time=25, value_wait=35, logit_scale_mu=logit_scale_mu),
        Travelers(number_traveler=20, trip_length=20, value_time=25, value_wait=35, logit_scale_mu=logit_scale_mu),
        Travelers(number_traveler=20, trip_length=10, value_time=25, value_wait=35, logit_scale_mu=logit_scale_mu),
        Travelers(number_traveler=20, trip_length=8, value_time=25, value_wait=35, logit_scale_mu=logit_scale_mu),
    ]

    if isinstance(tnc_capacities, list):
        tnc_capacities = {f"TNC{i+1}": cap for i, cap in enumerate(tnc_capacities)}

    tnc_services = []
    for idx, (name, capacity) in enumerate(tnc_capacities.items()):
        tnc_services.append(
            TNC(
                name=name,
                ASC=10,
                fare=2.5 - 0.5 * idx,
                detour_ratio=1.4,
                average_speed=40,
                average_veh_travel_dist_per_day=8 * 40 / 1000,
                capacity_ratio_to_MaaS=0.2,
                total_service_capacity=capacity,
                trip_length_per_traveler_type=[traveler.trip_length for traveler in travelers],
                value_waiting_time_per_traveler_type=[traveler.value_wait for traveler in travelers],
                cost_purchasing_capacity_TNC=0.35 - 0.05 * idx,
                operating_cost=0.300,
                lambda_T=0,
                logit_scale_mu=logit_scale_mu,
            )
        )

    mt = MT(
        ASC=0.0,
        fare=2,
        detour_ratio=1.5,
        average_speed=25,
        n_transfer_per_length=0.15,
        access_time=1 / 6,
        transit_time=1 / 12,
    )

    tnc_capacity_ratio_map = {tnc.name: tnc.capacity_ratio_to_MaaS for tnc in tnc_services}
    tnc_total_capacity_map = {tnc.name: tnc.total_service_capacity for tnc in tnc_services}
    ref_tnc = tnc_services[0]

    maas = MaaS(
        ASC=5,
        fare=1,
        share_TNC_per_traveler_type=[0.30 for _ in travelers],
        detour_ratio_TNC=ref_tnc.detour_ratio,
        average_speed_TNC=ref_tnc.average_speed,
        capacity_ratio_from_TNC=tnc_capacity_ratio_map,
        total_service_capacity_TNC=tnc_total_capacity_map,
        average_veh_travel_dist_per_day_TNC=ref_tnc.average_veh_travel_dist_per_day,
        cost_purchasing_capacity_TNC=ref_tnc.cost_purchasing_capacity_TNC,
        trip_length_per_traveler_type=ref_tnc.trip_length_per_traveler_type,
        value_travel_time_per_traveler_type=[traveler.value_time for traveler in travelers],
        value_waiting_time_per_traveler_type=ref_tnc.value_waiting_time_per_traveler_type,
        detour_ratio_MT=mt.detour_ratio,
        average_speed_MT=mt.average_speed,
        transit_time_MT=mt.transit_time,
        n_transfer_per_length_MT=mt.n_transfer_per_length,
        cost_purchasing_capacity_MT=4,
        lambda_M=0,
        logit_scale_mu=logit_scale_mu,
    )

    services = [*tnc_services, mt, maas]
    service_indices = {service.name: idx for idx, service in enumerate(services)}

    allocation = {service.name: [0] * len(travelers) for service in services}
    for type_i, traveler in enumerate(travelers):
        for service in services:
            allocation[service.name][type_i] += traveler.number_traveler / len(services)

    n_types = len(travelers)
    allocation_history = {service.name: [] for service in services}
    allocation_by_type = {service.name: [[] for _ in travelers] for service in services}

    for service in services:
        service.get_allocation(allocation)

    converged_at_day = None
    upper_level_updates = 0
    debug_snapshots = []
    gradient_history = []
    total_travelers = float(sum(traveler.number_traveler for traveler in travelers))
    upper_level_relative_tolerance = 0.01
    convergence_stability_window = 50
    stable_convergence_days = 0
    base_step_sizes_t = np.array([5e-4, 4e-5, 5e-5])             # was [1e-4, 1e-6, 1e-5]
    base_step_sizes_m = np.concatenate(([5e-4], np.full(n_types, 6e-5), [5e-5]))  # was [1e-4, 1e-5, 1e-5]
    update_direction_t = np.array([1.0, 1.0, -1.0])
    update_direction_m = np.ones_like(base_step_sizes_m)
    update_direction_m[-1] = -1.0
    ls_alpha0_t = 1.0
    ls_alpha0_m = 1.0
    ls_rho = 0.3
    ls_c = 1e-4
    ls_max_iter = 200
    ls_min_alpha = 1e-7

    for day in tqdm(range(number_days), desc="Simulation Progress", unit="day"):
        allocation = distribute_travelers(travelers, services)

        for t_idx in range(len(travelers)):
            for service in services:
                if day >= 1:
                    prev = allocation_by_type[service.name][t_idx][day - 1]
                    allocation[service.name][t_idx] = 0.999 * prev + 0.001 * allocation[service.name][t_idx]

        for service in services:
            service.get_allocation(allocation)

        allocation_history, allocation_by_type = store_allocations(
            day, travelers, services, allocation, allocation_history, allocation_by_type
        )

        if day >= 1:
            max_relative_change = max(
                abs(allocation_history[service.name][-1] - allocation_history[service.name][-2])
                / max(total_travelers, 1.0)
                for service in services
            )
        else:
            max_relative_change = np.inf

        if day >= 1 and max_relative_change <= upper_level_relative_tolerance:
            stable_convergence_days += 1
        else:
            stable_convergence_days = 0

        if day >= 1 and stable_convergence_days >= convergence_stability_window:
            if converged_at_day is None:
                converged_at_day = day
            tqdm.write(
                f"Lower level converged at Day {day + 1} "
                f"(stable for {convergence_stability_window} iterations)"
            )

            upper_level_updates += 1
            stable_convergence_days = 0

            utilities = compute_utilities(travelers, services)

            grad_t_by_name = {}
            params_t_by_name = {}
            auto_check_by_name = {}
            for tnc in tnc_services:
                params_t = np.array([tnc.fare, tnc.capacity_ratio_to_MaaS, tnc.lambda_T])
                grad_t = tnc.gradient_objective(
                    utilities,
                    maas,
                    service_index_T=service_indices[tnc.name],
                    service_index_M=service_indices["MaaS"],
                )
                params_t_by_name[tnc.name] = params_t
                grad_t_by_name[tnc.name] = grad_t

            if enable_gradient_checks:
                tnc_ref = tnc
                grad_auto_t = grad(
                    lambda p, tnc_obj=tnc_ref: tnc_obj.compute_objective_function(
                        p,
                        travelers,
                        services,
                        service_index_T=service_indices[tnc_obj.name],
                    )
                )(params_t)
                auto_check_by_name[tnc.name] = bool(np.allclose(grad_t, grad_auto_t, atol=1e-5))

            params_m = np.concatenate(([maas.fare], np.asarray(maas.share_TNC_per_traveler_type), [maas.lambda_M]))
            grad_maas = maas.gradient_objective(utilities, service_index_M=service_indices["MaaS"])
            maas_auto_check = None
            if enable_gradient_checks:
                grad_auto_m = grad(
                    lambda p: maas.compute_objective_function(
                        p,
                        travelers,
                        services,
                        service_index_M=service_indices["MaaS"],
                    )
                )(params_m)
                maas_auto_check = bool(np.allclose(grad_maas, grad_auto_m, atol=1e-5))

            grad_event = {
                "update_idx": int(upper_level_updates),
                "day": int(day + 1),
                "grad_maas_norm": float(np.linalg.norm(grad_maas)),
                "maas_autograd_match": maas_auto_check,
            }
            for tnc in tnc_services:
                grad_event[f"grad_{tnc.name}_norm"] = float(np.linalg.norm(grad_t_by_name[tnc.name]))
                grad_event[f"{tnc.name}_autograd_match"] = auto_check_by_name.get(tnc.name)

            if debug_enabled:
                probabilities_pre = compute_choice_probabilities(utilities, mu=logit_scale_mu)
                snapshot_pre = build_debug_snapshot(day, travelers, services, allocation, utilities, probabilities_pre)
                debug_snapshots.append(snapshot_pre)

            for tnc in tnc_services:
                signed_grad_t = grad_t_by_name[tnc.name] * update_direction_t
                direction_t = -signed_grad_t
                params_t_current = params_t_by_name[tnc.name]
                obj_curr_t = tnc.compute_objective_function(
                    params_t_current,
                    travelers,
                    services,
                    service_index_T=service_indices[tnc.name],
                )
                alpha_t = ls_alpha0_t
                for _ in range(ls_max_iter):
                    candidate_t = params_t_current + alpha_t * base_step_sizes_t * direction_t
                    candidate_t = project_tnc_params(candidate_t)
                    obj_next_t = tnc.compute_objective_function(
                        candidate_t,
                        travelers,
                        services,
                        service_index_T=service_indices[tnc.name],
                    )
                    if obj_next_t <= obj_curr_t + ls_c * alpha_t * float(np.dot(signed_grad_t, direction_t)):
                        break
                    alpha_t *= ls_rho
                    if alpha_t < ls_min_alpha:
                        break
                new_params_t = params_t_current + alpha_t * base_step_sizes_t * direction_t
                new_params_t = project_tnc_params(new_params_t)
                tnc.fare, tnc.capacity_ratio_to_MaaS, tnc.lambda_T = new_params_t
                grad_event[f"ls_alpha_{tnc.name}"] = float(alpha_t)
                param_den_t = float(np.linalg.norm(params_t_current)) + 1e-8
                grad_event[f"param_delta_rel_{tnc.name}"] = float(
                    np.linalg.norm(new_params_t - params_t_current) / param_den_t
                )

            signed_grad_m = grad_maas * update_direction_m
            direction_m = -signed_grad_m
            params_m_current = params_m
            obj_curr_m = maas.compute_objective_function(
                params_m_current,
                travelers,
                services,
                service_index_M=service_indices["MaaS"],
            )
            alpha_m = ls_alpha0_m
            for _ in range(ls_max_iter):
                candidate_m = params_m_current + alpha_m * base_step_sizes_m * direction_m
                candidate_m = project_maas_params(candidate_m, n_types=n_types)
                obj_next_m = maas.compute_objective_function(
                    candidate_m,
                    travelers,
                    services,
                    service_index_M=service_indices["MaaS"],
                )
                if obj_next_m <= obj_curr_m + ls_c * alpha_m * float(np.dot(signed_grad_m, direction_m)):
                    break
                alpha_m *= ls_rho
                if alpha_m < ls_min_alpha:
                    break
            new_params_m = params_m_current + alpha_m * base_step_sizes_m * direction_m
            new_params_m = project_maas_params(new_params_m, n_types=n_types)
            maas.fare = new_params_m[0]
            maas.share_TNC_per_traveler_type = np.array(new_params_m[1:1 + n_types])
            maas.lambda_M = new_params_m[1 + n_types]
            grad_event["ls_alpha_MaaS"] = float(alpha_m)
            param_den_m = float(np.linalg.norm(params_m_current)) + 1e-8
            grad_event["param_delta_rel_MaaS"] = float(
                np.linalg.norm(new_params_m - params_m_current) / param_den_m
            )

            gradient_history.append(grad_event)

            maas.capacity_ratio_from_TNC_by_name = {
                tnc.name: float(tnc.capacity_ratio_to_MaaS) for tnc in tnc_services
            }
            maas.total_service_capacity_TNC_by_name = {
                tnc.name: float(tnc.total_service_capacity) for tnc in tnc_services
            }
            maas.sync_tnc_pool_aliases()

            for tnc in tnc_services:
                tqdm.write(
                    f"[Update {upper_level_updates}] {tnc.name}: fare={tnc.fare:.4f}, "
                    f"cap_ratio={tnc.capacity_ratio_to_MaaS:.4f}, lambda={tnc.lambda_T:.6f}"
                )
            share_str = ", ".join(f"{v:.4f}" for v in maas.share_TNC_per_traveler_type)
            tqdm.write(f"[Update {upper_level_updates}] MaaS: fare={maas.fare:.4f}, share_TNC_per_type=[{share_str}], lambda={maas.lambda_M:.6f}")

    print("\n" + "=" * 80)
    print("SIMULATION SUMMARY")
    print("=" * 80)

    if converged_at_day is not None:
        print(f"\nConvergence reached at Day {converged_at_day + 1}")
        print(f"Upper-level optimization updates: {upper_level_updates}")
    else:
        print("\nMaximum iterations reached without convergence")

    print("\nFinal Allocation:")
    for service_name, allocation_vals in allocation.items():
        print(f"  {service_name}: {[round(v, 2) for v in allocation_vals]}")

    print("\nFinal TNC params:")
    for tnc in tnc_services:
        print(
            f"  {tnc.name}: fare={tnc.fare:.4f}, capacity_ratio_to_MaaS={tnc.capacity_ratio_to_MaaS:.4f}, "
            f"lambda_T={tnc.lambda_T:.6f}"
        )
    print("\nFinal MaaS params:")
    print(
        f"  fare={maas.fare:.4f}, share_TNC_per_type={[round(float(v), 4) for v in maas.share_TNC_per_traveler_type]}, "
        f"lambda_M={maas.lambda_M:.6f}"
    )

    financials = compute_operator_financials(travelers, tnc_services, maas, allocation)
    print("\nFinancial Results:")
    for name, vals in financials["tnc"].items():
        print(
            f"  {name}: Income=${vals['rider_revenue'] + vals['capacity_sale_revenue']:.2f}, "
            f"Cost=${vals['operating_cost']:.2f}, Profit=${vals['net_profit']:.2f}"
        )
    print(
        f"  MaaS: Income=${financials['maas']['rider_revenue']:.2f}, "
        f"Cost=${financials['maas']['tnc_capacity_purchase_cost'] + financials['maas']['mt_capacity_purchase_cost']:.2f}, "
        f"Profit=${financials['maas']['net_profit']:.2f}"
    )

    print("=" * 80 + "\n")

    results = {
        "tnc_capacities": {tnc.name: float(tnc.total_service_capacity) for tnc in tnc_services},
        "final_allocation": allocation,
        "tnc_params": {
            tnc.name: {
                "fare": float(tnc.fare),
                "capacity_ratio_to_MaaS": float(tnc.capacity_ratio_to_MaaS),
                "lambda_T": float(tnc.lambda_T),
            }
            for tnc in tnc_services
        },
        "maas_params": {
            "fare": float(maas.fare),
            "share_TNC_per_traveler_type": [float(v) for v in maas.share_TNC_per_traveler_type],
            "lambda_M": float(maas.lambda_M),
            "pooled_tnc_capacity_vkm": float(maas.get_pooled_tnc_capacity()),
        },
        "profits": {
            "tnc": {name: vals["net_profit"] for name, vals in financials["tnc"].items()},
            "maas": float(financials["maas"]["net_profit"]),
        },
    }

    os.makedirs(output_dir, exist_ok=True)

    if debug_enabled:
        store_debug_snapshots(output_dir=output_dir, snapshots=debug_snapshots)

    with open(os.path.join(output_dir, "final_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    with open(os.path.join(output_dir, "gradient_history.json"), "w", encoding="utf-8") as f:
        json.dump({"events": gradient_history}, f, indent=4)

    plot_total_allocations(
        services,
        allocation_history,
        number_days,
        save_path=os.path.join(output_dir, "total_allocation.png"),
    )

    plot_per_type_allocations(
        services,
        allocation_by_type,
        travelers,
        number_days,
        save_path=os.path.join(output_dir, "allocation_by_type.png"),
    )

    plot_gradient_evolution(
        gradient_history,
        [tnc.name for tnc in tnc_services],
        save_path=os.path.join(output_dir, "gradient_evolution.png"),
    )

    plot_param_delta_evolution(
        gradient_history,
        [tnc.name for tnc in tnc_services],
        save_path=os.path.join(output_dir, "param_delta_evolution.png"),
    )


if __name__ == "__main__":
    run_simulation(
        tnc_capacities={"TNC1": 1200, "TNC2": 1200},
        output_dir="./2-Results/v3_demo",
        number_days=1000,
        debug_enabled=True,
        enable_gradient_checks=True,
    )
