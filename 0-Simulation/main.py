from entities import Service, TNC, MT, MaaS, Travelers, distribute_travelers
# import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
import json
import os

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


def run_simulation(tnc_capacity: float, output_dir: str, number_days: int):
    # --------------------------
    # 0. Initialization
    # --------------------------
    number_days = number_days
    # --------------------------
    # 1. Define traveler groups
    # --------------------------
    travelers = [
        Travelers(number_traveler=200, trip_length=30, value_time=25, value_wait=50), # count, km, monetary unit/h, monetary unit/h
        Travelers(number_traveler=150, trip_length=10, value_time=30, value_wait=60),
        Travelers(number_traveler=100, trip_length=5, value_time=20, value_wait=40)
    ]

    # --------------------------
    # 2. Define services
    # --------------------------

    tnc = TNC(
        ASC=10.0, 
        fare=3, # monetary units per km
        detour_ratio=1.3, # 1.3 times the direct distance
        average_speed=40, # in km/h
        average_veh_travel_dist_per_day=8*40, # 320 km per veh per day
        capacity_ratio_to_MaaS=0.4, # TNC gives 40% of its capacity to MaaS
        total_service_capacity=tnc_capacity, # in veh * km per day
        trip_length_per_traveler_type=[traveler.trip_length for traveler in travelers], # km
        value_waiting_time_per_traveler_type=[traveler.value_wait for traveler in travelers], # monetary units per time
        cost_purchasing_capacity_TNC= 10, # monetary units per veh 
        operating_cost= 10, # monetary units per veh 
        lambda_T=1.0 # Lagrange multiplier for the capacity constraint [$/(veh·km)]
    )   

    mt = MT(
        ASC=0.0, 
        fare=9, # monetary units per segment (* n_transfer_per_length (eg. 0.3) = monetary units per km)
        detour_ratio=1.8,
        average_speed=15,
        n_transfer_per_length=0.3, # per km
        access_time=1/6, # hours
        transit_time=1/12 # hours
    )

    maas = MaaS(
        ASC=5, 
        fare=2, # additional maas operation cost * (...) monetary units per km 
        share_TNC=0.2, # share of TNC inside MaaS (first and last kilometers)
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
        cost_purchasing_capacity_MT=5, # MM unit ??
        lambda_M=1.0 # Lagrange multiplier for the capacity constraint [$/(veh·km)] 
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
    
    # --------------------------
    # 4. Simulation loop
    # --------------------------
    for day in range(number_days):
        print(f"\nDay {day + 1}/{number_days}")
        tnc.get_allocation(allocation)
        maas.get_allocation(allocation)
        allocation = distribute_travelers(travelers, services)
        for t_idx in range(len(travelers)):
            for service in services:
                # smooth allocations
                allocation[service.name][t_idx] = 0.99*allocation_by_type[service.name][t_idx][day-1]+ 0.01*allocation[service.name][t_idx] if day >=1 else allocation[service.name][t_idx]

        # Store allocations
        allocation_history, allocation_by_type = store_allocations(day, travelers, services, allocation, allocation_history, allocation_by_type) 
        
        # Check convergence of lower level and update upper level if converged
        if day >= 1 and all(np.all(np.abs(allocation_history[service.name][-1] - allocation_history[service.name][-2]) <= 0.0001) for service in services): # and check_gradients(travelers, services, utilities): not necessary to check gradients every time
                print("\nLower level converged.")

                # allocation
                #print(f"Allocation: {', '.join([f'{k}: {[round(v) for v in vals]}' for k, vals in allocation.items()])}")
                
                step_size = 1e-6

                # Compute utilities
                utilities = compute_utilities(travelers, services)

                # === TNC update ===
                tnc = next(service for service in services if service.name == "TNC")
                params_T = np.array([tnc.fare, tnc.capacity_ratio_to_MaaS, tnc.lambda_T])

                grad_tnc = tnc.gradient_objective(utilities, next(service for service in services if service.name == "MaaS"))

                new_params_T = params_T - step_size * grad_tnc
                new_params_T = project_tnc_params(new_params_T)

                tnc.fare, tnc.capacity_ratio_to_MaaS, tnc.lambda_T = new_params_T

                # === MaaS update ===
                maas = next(service for service in services if service.name == "MaaS")
                params_M = np.array([maas.fare, maas.share_TNC, maas.lambda_M])

                grad_maas = maas.gradient_objective(utilities)

                new_params_M = params_M - step_size * grad_maas
                new_params_M = project_maas_params(new_params_M) 

                maas.fare, maas.share_TNC, maas.lambda_M = new_params_M

                maas.capacity_ratio_from_TNC = tnc.capacity_ratio_to_MaaS # put this after because both service should update simultaneously 

                # qualitative informations
                print(f"Updated TNC params: fare={tnc.fare}, capacity_ratio_to_MaaS={tnc.capacity_ratio_to_MaaS}, lambda_T={tnc.lambda_T}")
                print(f"Updated MaaS params: fare={maas.fare}, share_TNC={maas.share_TNC}, lambda_M={maas.lambda_M}")

                # gradients and sizes
                #print(f"Gradients: grad_TNC={grad_tnc}, grad_MaaS={grad_maas}")
                #print(f"Gradient sizes: ||grad_TNC||={np.linalg.norm(grad_tnc)}, ||grad_MaaS||={np.linalg.norm(grad_maas)}")

    ###########################################################################
    ############################## END of the SIMULATION ######################
    ###########################################################################

    print("\nFinal allocation:")
    print(f"{', '.join([f'{k}: {[round(v) for v in vals]}' for k, vals in allocation.items()])}")

    print(f"Updated TNC params: fare={tnc.fare}, capacity_ratio_to_MaaS={tnc.capacity_ratio_to_MaaS}, lambda_T={tnc.lambda_T}")
    print(f"Updated MaaS params: fare={maas.fare}, share_TNC={maas.share_TNC}, lambda_M={maas.lambda_M}")
    
    # gradients
    print(f"Gradients: grad_TNC={grad_tnc}, grad_MaaS={grad_maas}")
    #size of the gradients
    print(f"Gradient sizes: ||grad_TNC||={np.linalg.norm(grad_tnc)}, ||grad_MaaS||={np.linalg.norm(grad_maas)}")
    # tnc profit
    income_tnc = np.sum([tnc.fare * allocation['TNC'][i] * tnc.detour_ratio * travelers[i].trip_length for i in range(len(travelers))]) + tnc.cost_purchasing_capacity_TNC * tnc.capacity_ratio_to_MaaS * tnc.total_service_capacity / tnc.average_veh_travel_dist_per_day
    outcome_tnc = tnc.operating_cost * tnc.total_service_capacity / tnc.average_veh_travel_dist_per_day
    print(f"\nTNC income: {income_tnc}, outcome: {outcome_tnc}, profit: {income_tnc - outcome_tnc}")
    # maas profit
    income_maas = np.sum([maas.fare * allocation['MaaS'][i] * travelers[i].trip_length for i in range(len(travelers))]) 
    outcome_maas = maas.cost_purchasing_capacity_TNC * maas.capacity_ratio_from_TNC * tnc.total_service_capacity / tnc.average_veh_travel_dist_per_day + maas.cost_purchasing_capacity_MT * (1 - maas.share_TNC) * np.sum(allocation['MaaS'])
    print(f"\nMaaS income: {income_maas}, outcome: {outcome_maas}, profit: {income_maas - outcome_maas}")

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
            "tnc_profit": income_tnc - outcome_tnc,
            "maas_profit": income_maas - outcome_maas,
        }
    }

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump(results, f, indent=4)

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
