from entities import Service, TNC, MT, MaaS, Travelers, distribute_travelers
# import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
import json
import os
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

def plot_total_allocations(services, allocation_history, number_days, save_path=None):
    """
    Description
    - Plot total allocations per service over time or store the image.

    Parameters
    - services: list of service objects used in the simulation.
    - allocation_history: number of travelers per service for each day (e.g. {"TNC": [10, 15, 15, ..., 15], ...}).
    - number_days: length of the simulation (number of days).
    - save_path: path to save the plot image. If None, the plot is shown instead.

    Output
    - Shows a plot or saves it to the specified path.
    """
    plt.figure(figsize=(8, 5))
    for service in services:
        plt.plot(range(number_days),
                 allocation_history[service.name],
                 label=service.name,
                 linewidth=2)
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
    - Plot allocations per traveler type for each service over time or store the image.

    Parameters
    - services: list of service objects used in the simulation.
    - allocation_by_type: number of travelers per service for each day split by type
                         (e.g. {"TNC": [[10, 14, ..., 15], [2, 3, ..., 3], [2, 3, ..., 3]], ...}).
    - travelers: list of traveler group objects used in the simulation.
    - number_days: length of the simulation (number of days).
    - save_path: path to save the plot image. If None, the plot is shown instead.

    Output
    - Shows a plot or saves it to the specified path.
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


def plot_gradient_evolution(gradient_history, save_path=None):
    """
    Description
    - Plot upper-level gradient norm evolution across updates.

    Parameters
    - gradient_history: list of dicts with keys: update_idx, day, grad_tnc_norm, grad_maas_norm.
    - save_path: path to save the plot image. If None, the plot is shown instead.

    Output
    - Shows a plot or saves it to the specified path.
    """
    if not gradient_history:
        return

    updates = [item["update_idx"] for item in gradient_history]
    grad_tnc_norm = [item["grad_tnc_norm"] for item in gradient_history]
    grad_maas_norm = [item["grad_maas_norm"] for item in gradient_history]

    plt.figure(figsize=(8, 5))
    plt.plot(updates, grad_tnc_norm, label="||grad_TNC||", linewidth=2)
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

def compute_utilities(travelers: list[Travelers], services: list[Service]) -> np.ndarray:
    """
    Description
    - Compute the utility matrix for all traveler types and services.

    Parameters
    - travelers: list of traveler group objects used in the simulation.
    - services: list of service objects used in the simulation.

    Output
    - utilities: 2D numpy array where utilities[i][j] is the utility for traveler type i using service j.
    """
    utilities = []
    for traveler in travelers:
        row = []
        for service in services:
            utility = service.compute_utility(
                traveler.trip_length, traveler.value_time, traveler.value_wait
            )
            row.append(utility)  
        utilities.append(row)
    return np.array(utilities)

def check_gradients(travelers: list[Travelers], services: list[Service], utilities: np.ndarray) -> bool:
    """
    Description
    - Verify the manually computed gradients against those computed using autograd.

    Parameters
    - travelers: list of traveler group objects used in the simulation.
    - services: list of service objects used in the simulation.
    - utilities: 2D numpy array where utilities[i][j] is the utility for traveler type i using service j.

    Output
    - True if all gradients match within a tolerance, False otherwise.
    """
    # ======== TNC GRADIENTS WITH AUTOGRAD ========
    tnc = next(service for service in services if service.name == "TNC")
    params_T = np.array([tnc.fare, tnc.capacity_ratio_to_MaaS, tnc.lambda_T])
    grad_tnc = grad(lambda p: tnc.compute_objective_function(p, travelers, services))(params_T) 

    manual_grad_tnc = tnc.gradient_objective(utilities, next(service for service in services if service.name == "MaaS"))
    if not np.allclose(grad_tnc, manual_grad_tnc, atol=1e-5):
        return False

    # ======== MAAS GRADIENTS WITH AUTOGRAD ========
    maas = next(service for service in services if service.name == "MaaS")
    params_M = np.array([maas.fare, maas.share_TNC, maas.lambda_M]) 
    grad_maas = grad(lambda p: maas.compute_objective_function(p, travelers, services))(params_M)

    manual_grad_maas = maas.gradient_objective(utilities)
    if not np.allclose(grad_maas, manual_grad_maas, atol=1e-5):
        return False

    return True

def store_allocations(day: int, travelers: list[Travelers], services: list[Service], allocation: dict[str, list[float]], allocation_history: dict[str, list[float]], allocation_by_type: dict[str, list[list[float]]]):
    """
    Description
    - Store allocations for each service and traveler type.

    Parameters
    - day: current day of the simulation.
    - travelers: list of traveler group objects used in the simulation.
    - services: list of service objects used in the simulation.
    - allocation: current allocation dictionary.

    Output
    - allocation_history: updated total allocations per service.
    - allocation_by_type: updated allocations per service and traveler type.
    """

    for service in services:
        total_travelers = sum(allocation[service.name])
        allocation_history[service.name].append(total_travelers)
        for t_idx in range(len(travelers)):
            if len(allocation_by_type[service.name][t_idx]) < day + 1:
                allocation_by_type[service.name][t_idx].append(allocation[service.name][t_idx])
            else:
                allocation_by_type[service.name][t_idx][day] = allocation[service.name][t_idx]

    return allocation_history, allocation_by_type

def project_tnc_params(params):
    fare, cap_ratio, lambda_T = params
    fare = max(fare, 0.0)
    cap_ratio = np.clip(cap_ratio, 0.0, 1.0)
    lambda_T = max(lambda_T, 0.0)
    return np.array([fare, cap_ratio, lambda_T])


def project_maas_params(params):
    fare, share_TNC, lambda_M = params
    fare = max(fare, 0.0)
    share_TNC = np.clip(share_TNC, 0.0, 1.0)
    lambda_M = max(lambda_M, 0.0)
    return np.array([fare, share_TNC, lambda_M])


def compute_choice_probabilities(utilities: np.ndarray) -> np.ndarray:
    """
    Description
    - Compute stabilized softmax choice probabilities from utility matrix.

    Parameters
    - utilities: shape (n_types, n_services) utility matrix.

    Output
    - Returns probabilities with the same shape as utilities.
    """
    stabilized = utilities - np.max(utilities, axis=1, keepdims=True)
    exp_utilities = np.exp(stabilized)
    return exp_utilities / np.sum(exp_utilities, axis=1, keepdims=True)


def compute_operator_financials(travelers: list[Travelers], tnc: TNC, maas: MaaS, allocation: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """
    Description
    - Compute detailed income/cost decomposition for TNC and MaaS.

    Output
    - Returns a nested dictionary with detailed operator economics.
    """
    n_types = len(travelers)

    tnc_rider_revenue = np.sum([
        tnc.fare * allocation["TNC"][i] * tnc.detour_ratio * travelers[i].trip_length
        for i in range(n_types)
    ])
    tnc_capacity_sale_revenue = (
        tnc.cost_purchasing_capacity_TNC
        * tnc.capacity_ratio_to_MaaS
        * tnc.total_service_capacity
        / tnc.average_veh_travel_dist_per_day
    )
    tnc_operating_cost = (
        tnc.operating_cost * tnc.total_service_capacity / tnc.average_veh_travel_dist_per_day
    )
    tnc_net_profit = tnc_rider_revenue + tnc_capacity_sale_revenue - tnc_operating_cost

    maas_rider_revenue = np.sum([
        maas.fare * allocation["MaaS"][i] * travelers[i].trip_length
        for i in range(n_types)
    ])
    maas_tnc_capacity_purchase_cost = (
        maas.cost_purchasing_capacity_TNC
        * maas.capacity_ratio_from_TNC
        * tnc.total_service_capacity
        / tnc.average_veh_travel_dist_per_day
    )
    maas_mt_capacity_purchase_cost = (
        maas.cost_purchasing_capacity_MT * (1 - maas.share_TNC) * np.sum(allocation["MaaS"])
    )
    maas_net_profit = (
        maas_rider_revenue - maas_tnc_capacity_purchase_cost - maas_mt_capacity_purchase_cost
    )

    return {
        "tnc": {
            "rider_revenue": float(tnc_rider_revenue),
            "capacity_sale_revenue": float(tnc_capacity_sale_revenue),
            "operating_cost": float(tnc_operating_cost),
            "net_profit": float(tnc_net_profit),
        },
        "maas": {
            "rider_revenue": float(maas_rider_revenue),
            "tnc_capacity_purchase_cost": float(maas_tnc_capacity_purchase_cost),
            "mt_capacity_purchase_cost": float(maas_mt_capacity_purchase_cost),
            "net_profit": float(maas_net_profit),
        },
    }


def build_debug_snapshot(
    day: int,
    travelers: list[Travelers],
    services: list[Service],
    allocation: dict[str, list[float]],
    utilities: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """
    Description
    - Build a streamlined debug snapshot for one upper-level pre-update event.
    """
    tnc = next(service for service in services if service.name == "TNC")
    mt = next(service for service in services if service.name == "MT")
    maas = next(service for service in services if service.name == "MaaS")

    trip_lengths = np.array([traveler.trip_length for traveler in travelers], dtype=float)

    tnc_demand_vkm = np.sum(trip_lengths * np.array(allocation["TNC"], dtype=float))
    maas_tnc_demand_vkm = maas.share_TNC * np.sum(
        trip_lengths * np.array(allocation["MaaS"], dtype=float)
    )

    tnc_capacity_for_tnc_vkm = (1 - tnc.capacity_ratio_to_MaaS) * tnc.total_service_capacity
    tnc_waiting_per_type = [
        float(np.sum(tnc.waiting_time(traveler.trip_length))) for traveler in travelers
    ]
    maas_waiting_per_type = [
        float(np.sum(maas.waiting_time(traveler.trip_length))) for traveler in travelers
    ]

    utility_decomposition = []
    for i, traveler in enumerate(travelers):
        per_service = {}
        for service in services:
            components = service.decompose_utility_components(
                trip_length=traveler.trip_length,
                value_time=traveler.value_time,
                value_wait=traveler.value_wait,
            )
            serialized = {k: float(np.sum(v)) for k, v in components.items()}

            if service.name == "MaaS":
                maas_modes = service.decompose_mode_components(
                    trip_length=traveler.trip_length,
                    value_time=traveler.value_time,
                    value_wait=traveler.value_wait,
                )
                serialized["mode_split"] = {
                    k: float(np.sum(v)) for k, v in maas_modes.items()
                }

            per_service[service.name] = serialized

        utility_decomposition.append({
            "traveler_type": i,
            "trip_length": float(traveler.trip_length),
            "value_time": float(traveler.value_time),
            "value_wait": float(traveler.value_wait),
            "services": per_service,
        })

    demand_by_service = {
        service.name: [float(v) for v in allocation[service.name]] for service in services
    }
    total_demand_by_service = {
        service.name: float(np.sum(allocation[service.name])) for service in services
    }

    choice_probabilities_by_type = []
    for i, probs in enumerate(np.asarray(probabilities)):
        choice_probabilities_by_type.append({
            "traveler_type": int(i + 1),
            "probabilities": [float(p) for p in probs],
        })

    return {
        "day": int(day),
        "update_day": int(day + 1),
        "allocation": {
            "total_by_service": total_demand_by_service,
            "by_service_and_type": demand_by_service,
        },
        "capacity": {
            "tnc": {
                "demand_vkm": float(tnc_demand_vkm),
                "capacity_for_tnc_vkm": float(tnc_capacity_for_tnc_vkm),
            },
            "maas": {
                "demand_vkm": float(maas_tnc_demand_vkm),
                "purchased_tnc_capacity_vkm": float(maas.capacity_ratio_from_TNC * maas.total_service_capacity_TNC),
            },
        },
        "waiting_times": {
            "tnc_mean_hr": float(np.mean(tnc_waiting_per_type)),
            "maas_mean_hr": float(np.mean(maas_waiting_per_type)),
        },
        "choice_probabilities": choice_probabilities_by_type,
        "utility_decomposition": utility_decomposition,
        "financials": compute_operator_financials(travelers, tnc, maas, allocation),
    }


def store_debug_snapshots(output_dir: str, snapshots: list[dict]) -> None:
    """
    Description
    - Store all upper-level pre-update debug snapshots to a single JSON file.
    """
    debug_path = os.path.join(output_dir, "debug_upper_level_preupdate.json")
    with open(debug_path, "w") as f:
        json.dump({"events": snapshots}, f, indent=4)


def run_simulation(
    tnc_capacity: float,
    output_dir: str,
    number_days: int,
    debug_enabled: bool = False,
):


    # --------------------------
    # 0. Initialization
    # --------------------------
    number_days = number_days
    # --------------------------
    # 1. Define traveler groups
    # --------------------------
    travelers = [
        Travelers(number_traveler=80000, trip_length=20, value_time=20, value_wait=30), # count, km, monetary unit/h, monetary unit/h
    ]

    # --------------------------
    # 2. Define services
    # --------------------------

    tnc = TNC(
        ASC=20, 
        fare=1.5, # monetary units per km
        detour_ratio=1.4, # 1.4 times the direct distance
        average_speed=40, # in km/h
        average_veh_travel_dist_per_day=8*40, # 320 km per veh per day
        capacity_ratio_to_MaaS=0.3, # TNC gives 30% of its capacity to MaaS
        total_service_capacity=tnc_capacity, # in veh * km per day
        trip_length_per_traveler_type=[traveler.trip_length for traveler in travelers], # km
        value_waiting_time_per_traveler_type=[traveler.value_wait for traveler in travelers], # monetary units per time
        cost_purchasing_capacity_TNC= 300, # monetary units per veh 
        operating_cost= 250, # monetary units per veh 
        lambda_T=0 # Lagrange multiplier for the capacity constraint [$/(veh·km)]
    )   

    mt = MT(
        ASC=0.0,
        fare=2,
        detour_ratio=1.5, # 1.5 times the direct distance
        average_speed=20, # in km/h
        n_transfer_per_length=0.2, # per km
        access_time=1/6, # hours
        transit_time=1/12 # hours
    )

    maas = MaaS(
        ASC=5, 
        fare=1, # additional maas operation cost * (...) monetary units per km 
        share_TNC=0.30, # share of TNC inside MaaS (first and last kilometers)
        detour_ratio_TNC=tnc.detour_ratio,
        average_speed_TNC=tnc.average_speed,
        capacity_ratio_from_TNC=tnc.capacity_ratio_to_MaaS,
        total_service_capacity_TNC=tnc.total_service_capacity,
        average_veh_travel_dist_per_day_TNC=tnc.average_veh_travel_dist_per_day,
        cost_purchasing_capacity_TNC=tnc.cost_purchasing_capacity_TNC, 
        trip_length_per_traveler_type=tnc.trip_length_per_traveler_type,
        value_travel_time_per_traveler_type=[traveler.value_time for traveler in travelers],
        value_waiting_time_per_traveler_type=tnc.value_waiting_time_per_traveler_type,
        detour_ratio_MT=mt.detour_ratio,
        average_speed_MT=mt.average_speed,
        transit_time_MT=mt.transit_time,
        n_transfer_per_length_MT=mt.n_transfer_per_length,
        cost_purchasing_capacity_MT=4, # MM unit ??
        lambda_M=0 # Lagrange multiplier for the capacity constraint [$/(veh·km)] 
        )

    services = [tnc, mt, maas]

    # --------------------------
    # 3. Uniform initial allocation
    # --------------------------
    allocation = {service.name: [0] * len(travelers) for service in services}

    for type_i, traveler in enumerate(travelers):
        for service in services:
            allocation[service.name][type_i] += traveler.number_traveler / len(services)

    # store allocation history
    allocation_history = {service.name: [] for service in services}
    allocation_by_type = {service.name: [[] for _ in travelers] for service in services}

    tnc.get_allocation(allocation)
    maas.get_allocation(allocation)

    # --------------------------
    # 4. Simulation loop
    # --------------------------
    converged_at_day = None
    upper_level_updates = 0
    debug_snapshots: list[dict] = []
    gradient_history: list[dict[str, float | int]] = []
    total_travelers = float(sum(traveler.number_traveler for traveler in travelers))
    upper_level_relative_tolerance = 6e-5
    
    for day in tqdm(range(number_days), desc="Simulation Progress", unit="day"):
        allocation = distribute_travelers(travelers, services)
        for t_idx in range(len(travelers)):
            for service in services:
                # smooth allocations
                allocation[service.name][t_idx] = 0.999*allocation_by_type[service.name][t_idx][day-1]+ 0.001*allocation[service.name][t_idx] if day >=1 else allocation[service.name][t_idx]

        tnc.get_allocation(allocation)
        maas.get_allocation(allocation)

        # Store allocations
        allocation_history, allocation_by_type = store_allocations(day, travelers, services, allocation, allocation_history, allocation_by_type) 

        # Check convergence of lower level using relative error and update upper level if converged
        if day >= 1:
            max_relative_change = max(
                abs(allocation_history[service.name][-1] - allocation_history[service.name][-2])
                / max(total_travelers, 1.0)
                for service in services
            )
        else:
            max_relative_change = np.inf

        if day >= 1 and max_relative_change <= upper_level_relative_tolerance: # and check_gradients(travelers, services, utilities): not necessary to check gradients every time
                # Track first convergence
                if converged_at_day is None:
                    converged_at_day = day
                    tqdm.write(f"\n✓ Lower level FIRST convergence at Day {day + 1}")
                else:
                    tqdm.write(f"✓ Lower level reconverged at Day {day + 1}")

                upper_level_updates += 1
                
                # Use parameter-specific step sizes: [fare, ratio/share, multiplier]
                # Smaller ratio/share steps help avoid oscillation near [0, 1] bounds.
                step_sizes_T = np.array([2e-10, 1e-10, 1e-5])
                step_sizes_M = np.array([2e-10, 1e-10, 1e-5])

                # Define update directions: Descent (-), Descent (-), Ascent (+)
                # Multiplying the step by [1, 1, -1] turns a subtraction into an addition for the 3rd term.
                update_direction = np.array([1.0, 1.0, -1.0])

                # Compute utilities (State A)
                utilities = compute_utilities(travelers, services)

                # === Get TNC & MaaS Initial Params & Manual Gradients ===
                tnc = next(service for service in services if service.name == "TNC")
                params_T = np.array([tnc.fare, tnc.capacity_ratio_to_MaaS, tnc.lambda_T])
                grad_tnc = tnc.gradient_objective(utilities, next(service for service in services if service.name == "MaaS"))

                maas = next(service for service in services if service.name == "MaaS")
                params_M = np.array([maas.fare, maas.share_TNC, maas.lambda_M])
                grad_maas = maas.gradient_objective(utilities)

                gradient_history.append({
                    "update_idx": int(upper_level_updates),
                    "day": int(day + 1),
                    "grad_tnc_norm": float(np.linalg.norm(grad_tnc)),
                    "grad_maas_norm": float(np.linalg.norm(grad_maas)),
                })
                
                # ========== GRADIENT VERIFICATION (Done BEFORE mutating objects) ==========
                #grad_tnc_auto = grad(lambda p: tnc.compute_objective_function(p, travelers, services))(params_T)
                #grad_maas_auto = grad(lambda p: maas.compute_objective_function(p, travelers, services))(params_M)
                #tqdm.write(f"  [Gradient Check] TNC  Manual: {grad_tnc}, Autograd: {grad_tnc_auto}, Match: {np.allclose(grad_tnc, grad_tnc_auto, atol=1e-5)}")
                #tqdm.write(f"  [Gradient Check] MaaS Manual: {grad_maas}, Autograd: {grad_maas_auto}, Match: {np.allclose(grad_maas, grad_maas_auto, atol=1e-5)}")
                # =========================================================================

                if debug_enabled:
                    probabilities_pre = compute_choice_probabilities(utilities)
                    snapshot_pre = build_debug_snapshot(
                        day=day,
                        travelers=travelers,
                        services=services,
                        allocation=allocation,
                        utilities=utilities,
                        probabilities=probabilities_pre,
                    )
                    debug_snapshots.append(snapshot_pre)

                # === APPLY UPDATES (Move to State B) ===
                new_params_T = params_T - step_sizes_T * grad_tnc * update_direction
                new_params_T = project_tnc_params(new_params_T)
                tnc.fare, tnc.capacity_ratio_to_MaaS, tnc.lambda_T = new_params_T

                new_params_M = params_M - step_sizes_M * grad_maas * update_direction
                new_params_M = project_maas_params(new_params_M) 
                maas.fare, maas.share_TNC, maas.lambda_M = new_params_M

                maas.capacity_ratio_from_TNC = tnc.capacity_ratio_to_MaaS
                
                # Print update info on every upper-level update
                tqdm.write(f"  [Update {upper_level_updates}] TNC:  fare={tnc.fare:.4f}, cap_ratio={tnc.capacity_ratio_to_MaaS:.4f}, λ_T={tnc.lambda_T:.6f}")
                tqdm.write(f"  [Update {upper_level_updates}] MaaS: fare={maas.fare:.4f}, share_TNC={maas.share_TNC:.4f}, λ_M={maas.lambda_M:.6f}")

    ###########################################################################
    ############################## END of the SIMULATION ######################
    ###########################################################################

    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)
    
    if converged_at_day is not None:
        print(f"\n✓ Convergence reached at Day {converged_at_day + 1}")
        print(f"  Upper-level optimization updates: {upper_level_updates}")
    else:
        print("\n⚠ Maximum iterations reached without convergence")
    
    print("\nFinal Allocation:")
    for service_name, allocation_vals in allocation.items():
        print(f"  {service_name}: {[round(v, 2) for v in allocation_vals]}")

    print(f"\nFinal TNC params:")
    print(f"  fare={tnc.fare:.4f}, capacity_ratio_to_MaaS={tnc.capacity_ratio_to_MaaS:.4f}, lambda_T={tnc.lambda_T:.6f}")
    print(f"\nFinal MaaS params:")
    print(f"  fare={maas.fare:.4f}, share_TNC={maas.share_TNC:.4f}, lambda_M={maas.lambda_M:.6f}")
    
    # Compute profits
    income_tnc = np.sum([tnc.fare * allocation['TNC'][i] * tnc.detour_ratio * travelers[i].trip_length for i in range(len(travelers))]) + tnc.cost_purchasing_capacity_TNC * tnc.capacity_ratio_to_MaaS * tnc.total_service_capacity / tnc.average_veh_travel_dist_per_day
    outcome_tnc = tnc.operating_cost * tnc.total_service_capacity / tnc.average_veh_travel_dist_per_day
    profit_tnc = income_tnc - outcome_tnc
    
    income_maas = np.sum([maas.fare * allocation['MaaS'][i] * travelers[i].trip_length for i in range(len(travelers))]) 
    outcome_maas = maas.cost_purchasing_capacity_TNC * maas.capacity_ratio_from_TNC * tnc.total_service_capacity / tnc.average_veh_travel_dist_per_day + maas.cost_purchasing_capacity_MT * (1 - maas.share_TNC) * np.sum(allocation['MaaS'])
    profit_maas = income_maas - outcome_maas
    
    print("\nFinancial Results:")
    print(f"  TNC:  Income=${income_tnc:.2f}, Cost=${outcome_tnc:.2f}, Profit=${profit_tnc:.2f}")
    print(f"  MaaS: Income=${income_maas:.2f}, Cost=${outcome_maas:.2f}, Profit=${profit_maas:.2f}")
    
    # Gradient information (only available if convergence was reached)
    if converged_at_day is not None:
        print("\nGradient Information (Upper Level):")
        print(f"  ||grad_TNC||  = {np.linalg.norm(grad_tnc):.6f}")
        print(f"  ||grad_MaaS|| = {np.linalg.norm(grad_maas):.6f}")
    
    print("="*80 + "\n")

    # Store final results in a JSON file
    results = {
        "tnc_capacity": tnc_capacity,
        "final_allocation": allocation,
        "tnc_params": {
            "fare": tnc.fare,
            "capacity_ratio_to_MaaS": tnc.capacity_ratio_to_MaaS,
            "lambda_T": tnc.lambda_T,
        },
        "maas_params": {
            "fare": maas.fare,
            "share_TNC": maas.share_TNC,
            "lambda_M": maas.lambda_M,
        },
        "profits": {
            "tnc_profit": float(profit_tnc),
            "maas_profit": float(profit_maas),
        }
    }

    os.makedirs(output_dir, exist_ok=True)

    if debug_enabled:
        store_debug_snapshots(output_dir=output_dir, snapshots=debug_snapshots)

    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    with open(os.path.join(output_dir, "gradient_history.json"), "w") as f:
        json.dump({"events": gradient_history}, f, indent=4)

    # Plot results 
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
        save_path=os.path.join(output_dir, "gradient_evolution.png"),
    )


if __name__ == "__main__":
    capacities = [4000, 8000, 32000]
    number_days = [150000, 3000, 3000]

    for i, cap in enumerate(capacities):
        print(f"\nRunning simulation for TNC capacity = {cap}")
        run_simulation(
            tnc_capacity=cap,
            output_dir=f"./2-Results/tnc_capacity_{cap}",
            number_days=number_days[i]
        )
