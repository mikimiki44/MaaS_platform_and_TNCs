"""Debug-snapshot and financial-reporting helpers for the simulation."""

import json
import os
import autograd.numpy as np

from entities import TNC


def compute_operator_financials(travelers, tnc_services, maas, allocation):
    """
    Description
    - Compute per-operator and MaaS profit/loss given the current allocation.

    Parameters
    - travelers: list of Travelers objects.
    - tnc_services: list of TNC service objects.
    - maas: MaaS service object.
    - allocation: dict mapping service name to per-class allocation list.

    Output
    - Dict with structure
      ``{"tnc": {<name>: {rider_revenue, capacity_sale_revenue, operating_cost, net_profit}, ...},
         "maas": {rider_revenue, tnc_capacity_purchase_cost, mt_capacity_purchase_cost, net_profit}}``.
    """
    n_types = len(travelers)
    tnc_financials = {}

    for tnc in tnc_services:
        rider_revenue = np.sum([
            tnc.fare * allocation[tnc.name][i] * tnc.detour_ratio * travelers[i].trip_length
            for i in range(n_types)
        ])
        capacity_sale_revenue = (
            tnc.cost_purchasing_capacity_TNC
            * tnc.capacity_ratio_to_MaaS
            * tnc.total_service_capacity
            / tnc.average_veh_travel_dist_per_day
        )
        operating_cost = tnc.operating_cost * tnc.total_service_capacity / tnc.average_veh_travel_dist_per_day
        net_profit = rider_revenue + capacity_sale_revenue - operating_cost

        tnc_financials[tnc.name] = {
            "rider_revenue": float(rider_revenue),
            "capacity_sale_revenue": float(capacity_sale_revenue),
            "operating_cost": float(operating_cost),
            "net_profit": float(net_profit),
        }

    maas_rider_revenue = np.sum([
        maas.fare * allocation["MaaS"][i] * travelers[i].trip_length
        for i in range(n_types)
    ])
    maas_tnc_purchase_cost = 0.0
    for tnc in tnc_services:
        maas_tnc_purchase_cost += (
            maas.cost_purchasing_capacity_TNC
            * maas.capacity_ratio_from_TNC_by_name.get(tnc.name, 0.0)
            * tnc.total_service_capacity
            / tnc.average_veh_travel_dist_per_day
        )

    maas_mt_purchase_cost = maas.cost_purchasing_capacity_MT * np.sum(
        (1 - np.asarray(maas.share_TNC_per_traveler_type)) * np.asarray(allocation["MaaS"])
    )
    maas_net_profit = maas_rider_revenue - maas_tnc_purchase_cost - maas_mt_purchase_cost

    return {
        "tnc": tnc_financials,
        "maas": {
            "rider_revenue": float(maas_rider_revenue),
            "tnc_capacity_purchase_cost": float(maas_tnc_purchase_cost),
            "mt_capacity_purchase_cost": float(maas_mt_purchase_cost),
            "net_profit": float(maas_net_profit),
        },
    }


def decompose_utility_components(service, trip_length, value_time, value_wait, traveler_type_idx=None):
    """
    Description
    - Break the utility of ``service`` for a given trip into ASC, fare-disutility,
      travel-time disutility, waiting-time disutility, and the total. Used by
      the debug viewer's utility-decomposition table.

    Parameters
    - service: a Service instance (TNC, MT, or MaaS).
    - trip_length: trip distance [km].
    - value_time: value of in-vehicle time [$/hr].
    - value_wait: value of waiting time [$/hr].
    - traveler_type_idx: class index, required for class-dependent services (MaaS).

    Output
    - Dict with keys ``ASC``, ``fare_disutility``, ``travel_time_disutility``,
      ``waiting_time_disutility``, ``total_utility`` (all scalars).
    """
    fare = service.trip_fare(trip_length)
    time = service.trip_time(trip_length, traveler_type_idx=traveler_type_idx)
    wait = service.waiting_time(trip_length, traveler_type_idx=traveler_type_idx)
    return {
        "ASC": float(np.sum(service.ASC)),
        "fare_disutility": float(np.sum(fare)),
        "travel_time_disutility": float(np.sum(value_time * time)),
        "waiting_time_disutility": float(np.sum(value_wait * wait)),
        "total_utility": float(np.sum(service.ASC - fare - value_time * time - value_wait * wait)),
    }


def decompose_mode_components(maas, trip_length, value_time, value_wait, traveler_type_idx=None):
    """
    Description
    - Split the MaaS in-vehicle time and waiting time into their TNC-leg and
      MT-leg components, weighted by ``alpha_i = share_TNC_per_traveler_type[i]``.
      Used by the debug viewer's "MaaS split" rows.

    Parameters
    - maas: the MaaS service instance.
    - trip_length: trip distance [km].
    - value_time: value of in-vehicle time [$/hr].
    - value_wait: value of waiting time [$/hr].
    - traveler_type_idx: class index for selecting ``alpha_i``.

    Output
    - Dict with per-leg time, wait, and per-leg monetary disutility components.
    """
    A = 2.5
    sensitivity_param = 0.5
    share_tnc = float(maas.share_TNC_per_traveler_type[traveler_type_idx])
    vacant = maas.find_vacant_veh_available()
    vacant = np.where(vacant <= 0, 1e-6, vacant)

    time_tnc = maas.detour_ratio_TNC * share_tnc * trip_length / maas.average_speed_TNC
    time_mt = maas.detour_ratio_MT * (1 - share_tnc) * trip_length / maas.average_speed_MT

    wait_tnc = A * (vacant ** (-sensitivity_param))
    wait_mt = maas.transit_time_MT * (
        maas.n_transfer_per_length_MT * (1 - share_tnc) * trip_length
    )

    return {
        "trip_fare": float(np.sum(maas.trip_fare(trip_length))),
        "time_tnc": float(np.sum(time_tnc)),
        "time_mt": float(np.sum(time_mt)),
        "wait_tnc": float(np.sum(wait_tnc)),
        "wait_mt": float(np.sum(wait_mt)),
        "travel_time_disutility_tnc": float(np.sum(value_time * time_tnc)),
        "travel_time_disutility_mt": float(np.sum(value_time * time_mt)),
        "waiting_disutility_tnc": float(np.sum(value_wait * wait_tnc)),
        "waiting_disutility_mt": float(np.sum(value_wait * wait_mt)),
    }


def build_debug_snapshot(day, travelers, services, allocation, utilities, probabilities):
    """
    Description
    - Build one pre-update debug snapshot containing everything the interactive
      viewer needs: per-service totals, per-TNC capacity / demand / waiting,
      pooled-MaaS capacity, mean waiting times, structured choice probabilities,
      per-class utility decomposition (with MaaS TNC-vs-MT mode split), and
      operator financials.

    Parameters
    - day: zero-indexed day at which the snapshot is taken.
    - travelers: list of Travelers objects.
    - services: list of Service objects (TNCs, MT, MaaS).
    - allocation: dict mapping service name to per-class allocation list.
    - utilities: 2-D array of shape (n_classes, n_services) of utility values.
    - probabilities: 2-D array of softmax choice probabilities matching ``utilities``.

    Output
    - Dict ready to be serialised into the debug JSON event list (rich schema).
    """
    tnc_services = [service for service in services if isinstance(service, TNC)]
    maas = next(service for service in services if service.name == "MaaS")

    trip_lengths = np.array([traveler.trip_length for traveler in travelers], dtype=float)
    share_tnc_per_type = np.asarray(maas.share_TNC_per_traveler_type, dtype=float)
    maas_allocation = np.array(allocation["MaaS"], dtype=float)

    by_service_and_type = {
        service.name: [float(v) for v in allocation[service.name]] for service in services
    }
    total_by_service = {
        service.name: float(np.sum(allocation[service.name])) for service in services
    }

    # ---- per-TNC capacity, demand, and mean waiting time ----
    tnc_capacity_block = {}
    tnc_waiting_per_operator = {}
    for tnc in tnc_services:
        demand_vkm = float(np.sum(trip_lengths * np.array(allocation[tnc.name], dtype=float)))
        capacity_for_tnc_vkm = float((1 - tnc.capacity_ratio_to_MaaS) * tnc.total_service_capacity)
        sold_to_maas_vkm = float(tnc.capacity_ratio_to_MaaS * tnc.total_service_capacity)
        tnc_capacity_block[tnc.name] = {
            "capacity_ratio_to_maas": float(tnc.capacity_ratio_to_MaaS),
            "total_capacity_vkm": float(tnc.total_service_capacity),
            "demand_vkm": demand_vkm,
            "capacity_for_tnc_vkm": capacity_for_tnc_vkm,
            "sold_to_maas_vkm": sold_to_maas_vkm,
        }
        tnc_waiting_per_operator[tnc.name] = float(np.mean([
            float(np.sum(tnc.waiting_time(traveler.trip_length, traveler_type_idx=i)))
            for i, traveler in enumerate(travelers)
        ]))

    # ---- pooled MaaS demand and capacity ----
    maas_tnc_demand_vkm_by_type = share_tnc_per_type * trip_lengths * maas_allocation
    maas_block = {
        "demand_vkm": float(np.sum(maas_tnc_demand_vkm_by_type)),
        "demand_vkm_by_type": [float(v) for v in maas_tnc_demand_vkm_by_type],
        "share_tnc_per_type": [float(v) for v in share_tnc_per_type],
        "purchased_tnc_capacity_vkm": float(maas.get_pooled_tnc_capacity()),
    }

    maas_waiting_mean_hr = float(np.mean([
        float(np.sum(maas.waiting_time(traveler.trip_length, traveler_type_idx=i)))
        for i, traveler in enumerate(travelers)
    ]))

    waiting_times = {f"{name}_mean_hr": val for name, val in tnc_waiting_per_operator.items()}
    waiting_times["maas_mean_hr"] = maas_waiting_mean_hr

    # ---- per-class utility decomposition (with MaaS mode-split) ----
    utility_decomposition = []
    for i, traveler in enumerate(travelers):
        per_service = {}
        for service in services:
            components = decompose_utility_components(
                service,
                trip_length=traveler.trip_length,
                value_time=traveler.value_time,
                value_wait=traveler.value_wait,
                traveler_type_idx=i,
            )
            if service.name == "MaaS":
                components["mode_split"] = decompose_mode_components(
                    maas,
                    trip_length=traveler.trip_length,
                    value_time=traveler.value_time,
                    value_wait=traveler.value_wait,
                    traveler_type_idx=i,
                )
            per_service[service.name] = components

        utility_decomposition.append({
            "traveler_type": i,
            "trip_length": float(traveler.trip_length),
            "value_time": float(traveler.value_time),
            "value_wait": float(traveler.value_wait),
            "services": per_service,
        })

    # ---- structured choice probabilities ----
    service_order = [service.name for service in services]
    choice_probabilities_by_type = []
    for i, probs in enumerate(np.asarray(probabilities)):
        choice_probabilities_by_type.append({
            "traveler_type": int(i + 1),
            "service_order": service_order,
            "probabilities": [float(p) for p in probs],
        })

    return {
        "day": int(day),
        "update_day": int(day + 1),
        "service_order": service_order,
        "allocation": {
            "total_by_service": total_by_service,
            "by_service_and_type": by_service_and_type,
        },
        "capacity": {
            "tnc_pool": tnc_capacity_block,
            "maas": maas_block,
            "maas_pooled_tnc_capacity_vkm": float(maas.get_pooled_tnc_capacity()),
            "maas_share_tnc_per_type": [float(v) for v in share_tnc_per_type],
        },
        "waiting_times": waiting_times,
        "choice_probabilities": choice_probabilities_by_type,
        "utility_decomposition": utility_decomposition,
        "financials": compute_operator_financials(travelers, tnc_services, maas, allocation),
    }


def store_debug_snapshots(output_dir, snapshots):
    """
    Description
    - Serialise the list of debug snapshots to ``debug_upper_level_preupdate.json``
      inside ``output_dir``.

    Parameters
    - output_dir: directory to write the JSON file into.
    - snapshots: list of dicts (typically produced by ``build_debug_snapshot``).
    """
    debug_path = os.path.join(output_dir, "debug_upper_level_preupdate.json")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump({"events": snapshots}, f, indent=4)
