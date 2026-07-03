from abc import ABC, abstractmethod
from typing import List, Dict, Optional
# import numpy as np
import autograd.numpy as np


# --------------------------
# Base Service Class
# --------------------------

class Service(ABC):
    """
    Description
    - Abstract base class representing a generic transportation service.
    """

    def __init__(self, name: str):
        """
        Description
        - Initialize the service with its name.

        Parameters
        - name: string name of the service (e.g., 'TNC', 'MT', 'MaaS').

        Output
        - Create an instance of the Service class (abstract).
        """
        self.name = name

    @abstractmethod
    def trip_fare(self, trip_length: float) -> float:
        """
        Description
        - Compute total fare for a trip of given length.

        Parameters
        - trip_length: trip distance [km].

        Output
        - Returns monetary fare for that trip [$].
        """
        raise NotImplementedError

    @abstractmethod
    def trip_time(self, trip_length: float, traveler_type_idx: Optional[int] = None) -> float:
        """
        Description
        - Compute total travel time for a trip of given length.

        Parameters
        - trip_length: trip distance [km].

        Output
        - Returns expected in-vehicle travel time [hr].
        """
        raise NotImplementedError

    @abstractmethod
    def waiting_time(self, trip_length: float, traveler_type_idx: Optional[int] = None) -> float:
        """
        Description
        - Compute expected waiting time for a trip of given length.

        Parameters
        - trip_length: trip distance [km].

        Output
        - Returns expected waiting time [hr].
        """
        raise NotImplementedError

    def compute_utility(
        self,
        trip_length: float,
        value_time: float,
        value_wait: float,
        traveler_type_idx: Optional[int] = None,
    ) -> float:
        """
        Description
        - Generic utility computation.

        Parameters
        - trip_length: trip distance [km].
        - value_time: value of time [$/hr].
        - value_wait: value of waiting time [$/hr].

        Output
        - Returns the utility function [$].
        """
        fare = self.trip_fare(trip_length)
        time = self.trip_time(trip_length, traveler_type_idx=traveler_type_idx)
        wait = self.waiting_time(trip_length, traveler_type_idx=traveler_type_idx)
        return np.sum(self.ASC - fare - value_time * time - value_wait * wait) # there is no sum here, but np.sum works with both floats and ArrayBox scalars, and always returns a scalar, which autograd is happy with.

    def get_allocation(self, allocation: dict[str, list[float]]) -> None:
        """
        Description
        - Provide the service with the current allocation (demand) per traveler type for each service.

        Parameters
        - allocation: dict mapping service name -> list of counts per traveler type [veh]. # MM : disscuss unit
            Example: {"TNC": [n0, n1, n2], "MT": [..], "MaaS": [...]}

        Output
        - Sets `self.demand_per_traveler_type` with the last allocation available from the simulation in a dict of array.
        """
        self.demand_per_traveler_type = {
            k: np.array(v, dtype=float) for k, v in allocation.items()
        }
        return


# --------------------------
# TNC Service
# --------------------------

class TNC(Service):
    """
    Description
    - Transportation Network Company (TNC) service model, e.g., ride-hailing.

    Parameters
    - Mother class Service.
    """

    def __init__(self,
        ASC: float,
        fare: float,
        detour_ratio: float,
        average_speed: float,
        average_veh_travel_dist_per_day: float,
        capacity_ratio_to_MaaS: float,
        total_service_capacity: float,
        trip_length_per_traveler_type: list[float],
        value_waiting_time_per_traveler_type: list[float],
        cost_purchasing_capacity_TNC: float,
        operating_cost: float,
        lambda_T: float,
        name: str = "TNC",
        logit_scale_mu: float = 1.0):
        """
        Description
        - Initialize a TNC service model.

        Attributes 
        - ASC: alternative specific constant to compute utility in monetary unit [$].
        - fare: monetary fare per distance [$/km].
        - detour_ratio: multiplicative factor for distance due to detours [-].
        - average_speed: average vehicle speed in [km/h].
        - average_veh_travel_dist_per_day: km covered by a single vehicle per day [km].
        - capacity_ratio_to_MaaS: fraction of TNC capacity allocated to MaaS service [-].
        - total_service_capacity: total TNC service capacity in veh·km per day [veh·km].
        - trip_length_per_traveler_type: trip length in km for each traveler type stored in a list [km].
        - value_waiting_time_per_traveler_type: value of waiting time in monetary unit for each traveler group stored in a list [$/hr].
        - cost_purchasing_capacity_TNC:  price per veh*km to sell it to MaaS operator [$/(veh·km)].
        - operating_cost: operating cost per capacity units [$/(veh·km)].
        - lambda_T: Lagrange multiplier for the capacity constraint [$/(veh·km)].

        Attributes (initialized later)
        - demand_per_traveler_type: dict mapping service name -> list of counts per traveler type [veh].
                                    Example: {"TNC": [n0, n1, n2], "MT": [..], "MaaS": [...]}
        - vacant_veh_available: number of vacant vehicules (dedicated to TNC) in the fleet [veh].

        Output
        - Create a class TNC and initialize all the attributes.
        """
        super().__init__(name=name)

        self.ASC = ASC
        self.fare = fare
        self.detour_ratio = detour_ratio
        self.average_speed = average_speed
        self.average_veh_travel_dist_per_day = average_veh_travel_dist_per_day
        self.capacity_ratio_to_MaaS = capacity_ratio_to_MaaS
        self.total_service_capacity = total_service_capacity
        
        # Parameters for objective
        self.cost_purchasing_capacity_TNC = cost_purchasing_capacity_TNC         
        self.operating_cost = operating_cost   
        self.lambda_T: float = lambda_T
        
        self.trip_length_per_traveler_type: list[float] = trip_length_per_traveler_type
        self.demand_per_traveler_type: dict[str, list[float]] | None = None
        self.value_waiting_time_per_traveler_type: list[float] = value_waiting_time_per_traveler_type
        self.logit_scale_mu: float = logit_scale_mu

    def trip_fare(self, trip_length: float) -> float:
        """
        Description
        - Compute TNC fare for a given trip length.

        Parameters
        - trip_length: trip distance [km].
        
        Output
        - Returns fare for that trip [$].
        """
        return self.fare * self.detour_ratio * trip_length 

    def trip_time(self, trip_length: float, traveler_type_idx: Optional[int] = None) -> float:
        """
        Description
        - Compute expected in-vehicle travel time.

        Parameters
        - trip_length: trip distance [km].
        
        Output
        - Returns time [hr].
        """
        return self.detour_ratio * trip_length / self.average_speed

    def waiting_time(self, trip_length = None, traveler_type_idx: Optional[int] = None) -> float:
        """
        Description
        - Compute expected waiting time using empirical matching model.

        Internal variables
        - A: exogenous matching factor, A = 2.5 (Zhou et al. (2022), Competition and third-party platform integration in ride-sourcing
                    markets. Transportation Research Part B: Methodological, 159, 76-103.)
        - sensitivity_param = 0.5 (elasticity w.r.t. vacant vehicles); typical value 0.5 for regular e-hailing services without
            passenger competition in the matching process.

        Output
        - Returns expected waiting time based on an empirical formula (hours).
        """
        A = 2.5
        sensitivity_param = 0.5
        vacant_veh_available = self.find_vacant_veh_available()
        return A * (vacant_veh_available ** (-sensitivity_param))

    def find_vacant_veh_available(self) -> float:
        """
        Description
        - Compute number of idle vehicles in the TNC fleet.

        Internal variables
        -total_demand: computed as sum over traveler types of (trip_length_i * allocated_travelers_i) [veh·km]
        
        Output
        - Returns the number of vacant vehicles based on the TNC fleet capacity alocated to TNC [veh].
        """
        total_demand = np.sum(
            np.array(self.trip_length_per_traveler_type)
            * np.array(self.demand_per_traveler_type[self.name])
        )
        vacant_veh_available = np.array(((1 - self.capacity_ratio_to_MaaS) * self.total_service_capacity - total_demand) / self.average_veh_travel_dist_per_day)
        vacant_veh_available = np.where(
            vacant_veh_available <= 0,
            1e-6,
            vacant_veh_available
        ) # safer for autograd
        return vacant_veh_available
    
    def compute_objective_function(self, params: list[float], travelers: List['Travelers'], services: List[Service], service_index_T: int = 0) -> float:
        """
        Description
        - Compute the TNC objective function (Lagrangian formulation). See calculation details in report.

        Parameters
        - params: list of operational variables [fare, capacity_ratio_to_MaaS, lambda_T].
        - travelers: list of Traveler objects in the simulation.
        - services: list of Service objects in the simulation.
        - service_index_T: index of the TNC column in U (default 0).

        Internal variables
        - U: shape (i_types, m_services) of utility values U_im for each
            traveler type i and service m.
        - l: array of trip lengths per traveler type (l_i) [km].
        - Q: array of total travelers per type aggregated across services (Q_i) [veh].
        - P: softmax probabilities over services from U (shape n_types x n_services).
        - P_iT: column vector of probabilities of choosing TNC for each traveler type.
        - sum_l_PiT_Qi: sum_i l_i * P_iT * Q_i (demand assigned to TNC) [veh*km].
        - sum_PiT_Qi: sum_i P_iT * Q_i (number of travelers choosing TNC weighted by Q) [veh].

        Objective terms (economic interpretation):
        - term1: TNC fare revenue (negative cost here because we minimize cost-like objective) [$].
        - term2: cost of purchasing capacity allocated to MaaS from TNC [$].
        - term3: operating cost of running the TNC fleet [$].
        - term4: Lagrangian term enforcing vehicle capacity constraint (lambda_T) [$].

        Output
        - Returns a scalar value of the objective [$].
        """
        # save originals
        fare0 = self.fare
        y0 = self.capacity_ratio_to_MaaS
        lam0 = self.lambda_T
        maas = next(service for service in services if service.name == "MaaS")
        y0_M = maas.capacity_ratio_from_TNC_by_name.get(self.name, 0.0)

        # compute objective with new params
        self.fare, self.capacity_ratio_to_MaaS, self.lambda_T = params

        maas.capacity_ratio_from_TNC_by_name[self.name] = self.capacity_ratio_to_MaaS
        maas.sync_tnc_pool_aliases()

        U = compute_utility_matrix(travelers, services)
        l = np.asarray(self.trip_length_per_traveler_type)
        Q = np.sum(list(self.demand_per_traveler_type.values()), axis=0)
        P = _logit_probabilities(U, self.logit_scale_mu)
        P_iT = P[:, service_index_T]
        sum_l_PiT_Qi = np.sum(l * P_iT * Q)
        sum_PiT_Qi   = np.sum(P_iT * Q)
        term1 = -self.fare * self.detour_ratio * sum_l_PiT_Qi
        term2 = -self.cost_purchasing_capacity_TNC * self.capacity_ratio_to_MaaS * self.total_service_capacity / self.average_veh_travel_dist_per_day
        term3 = self.operating_cost * self.total_service_capacity / self.average_veh_travel_dist_per_day
        term4 = self.lambda_T * (
            sum_l_PiT_Qi -
            (1 - self.capacity_ratio_to_MaaS) * self.total_service_capacity
        )

        # restore originals
        self.fare = fare0
        self.capacity_ratio_to_MaaS = y0
        self.lambda_T = lam0
        maas.capacity_ratio_from_TNC_by_name[self.name] = y0_M
        maas.sync_tnc_pool_aliases()

        return term1 + term2 + term3 + term4

    def gradient_objective(
            self,
            U: np.ndarray,
            maas: Service,
            service_index_T: int = 0,
            service_index_M: int = 2
        ) -> np.ndarray:
        """
        Description
        - Compute gradient of TNC objective w.r.t. operational variables (self.fare, self.capacity_ratio_to_MaaS and self.lambda_T).

        Parameters
        - U: shape (n_types, n_services) of utility values U_im for each
            traveler type i and service m (monetary units / utility scale).
        - maas: MaaS service object to access its find_vacant_veh_available() method.
        - service_index_T: index of the TNC column in U (default 0).
        - service_index_M: index of the MaaS column in U (default 2).

        Internal variables
        - l: array of trip lengths per traveler type (l_i) [km].
        - Q: array of total travelers per type aggregated across services (Q_i) [veh].
        - P: softmax probabilities over services from U (shape n_types x n_services).
        - P_iT / P_iM: column vectors of choice probabilities for TNC and MaaS.
        - dUdf: partial derivative of utility w.r.t. fare (vector).
        - dUdy: partial derivative of utility w.r.t. capacity_ratio_to_MaaS (vector).
        - sum_l_PiT_Qi: sum_i l_i * P_iT * Q_i (demand assigned to TNC) [veh*km].
        - sum_PiT_Qi: sum_i P_iT * Q_i (number of travelers choosing TNC weighted by Q) [veh].
        - A: exogenous matching factor, A = 2.5 (Zhou et al. (2022), Competition and third-party platform integration in ride-sourcing
            markets. Transportation Research Part B: Methodological, 159, 76-103.)
        - s: sensitivity parameter (elasticity w.r.t. vacant vehicles), typical value 0.5 for regular e-hailing services without
            passenger competition in the matching process.

        Output
        - Returns the gradient vector [dObj/d_fare, dObj/d_capacity_ratio_to_MaaS, dObj/d_lambda_T].
        """
        mu = self.logit_scale_mu
        l = np.asarray(self.trip_length_per_traveler_type)
        Q = np.sum(list(self.demand_per_traveler_type.values()), axis=0)
        P = _logit_probabilities(U, mu)

        P_iT = P[:, service_index_T]
        P_iM = P[:, service_index_M]

        dUdf = -self.detour_ratio * l

        A, s =2.5, 0.5
        vacant = self.find_vacant_veh_available()

        dUTdy = - np.asarray(self.value_waiting_time_per_traveler_type) * s * A * (vacant)**(-(s + 1)) * (self.total_service_capacity / self.average_veh_travel_dist_per_day)


        vacant = maas.find_vacant_veh_available() # tricks: use MaaS vacant vehicles
        dUMdy = np.asarray(self.value_waiting_time_per_traveler_type) * s * A * (vacant)**(-(s + 1)) * (self.total_service_capacity / self.average_veh_travel_dist_per_day)

        sum_l_PiT_Qi = np.sum(l * P_iT * Q)
        sum_PiT_Qi = np.sum(P_iT * Q)

        # Chain-rule sums through P_iT (each contains an explicit factor mu, the logit scale).
        chain_f = mu * np.sum(l * Q * P_iT * (1 - P_iT) * dUdf)
        chain_y = mu * np.sum(l * Q * P_iT * ((1 - P_iT) * dUTdy - P_iM * dUMdy))

        # ========= GRAD w.r.t. f_T ==========
        grad_fT = (
            -self.detour_ratio * sum_l_PiT_Qi
            - self.fare * self.detour_ratio * chain_f
            + self.lambda_T * chain_f)

        # ========= GRAD w.r.t. y_T ==========
        grad_yT = (
            -self.fare * self.detour_ratio * chain_y
            - self.cost_purchasing_capacity_TNC * self.total_service_capacity / self.average_veh_travel_dist_per_day
            + self.lambda_T * chain_y
            + self.lambda_T * self.total_service_capacity)

        # ========= GRAD w.r.t. λ_T ==========
        grad_lambdaT = (sum_l_PiT_Qi - (1 - self.capacity_ratio_to_MaaS) * self.total_service_capacity) 
        
        return np.array([grad_fT, grad_yT, grad_lambdaT])

# --------------------------
# Mass Transit Service
# --------------------------

class MT(Service):
    """
    Description
    - Mass Transit (MT) service model.

    Parameters
    - Mother class Service.
    """

    def __init__(self,
            ASC: float,
            fare: float,
            n_transfer_per_length: float,
            detour_ratio: float,
            average_speed: float,
            access_time: float,
            transit_time: float
        ):
        """
        Description  
        - Initialize a Mass Transit (MT) service model.

        Attributes  
        - ASC: alternative specific constant to compute utility in monetary unit [$].  
        - fare: monetary fare per segment [$/seg].  
        - n_transfer_per_length: average number of transfers per unit trip length [-/km].  
        - detour_ratio: multiplicative factor for distance due to detours [-].  
        - average_speed: average vehicle speed [km/h].  
        - access_time: average access time to reach transit service [hr].  
        - transit_time: average in-vehicle transit time for a trip [hr].

        Output  
        - Create a class MT and initialize all the attributes.
        """
        super().__init__(name="MT")

        self.ASC = ASC
        self.fare = fare
        self.n_transfer_per_length = n_transfer_per_length
        self.detour_ratio = detour_ratio 
        self.average_speed = average_speed
        self.access_time = access_time
        self.transit_time = transit_time

    def trip_fare(self, trip_length: float) -> float:
        """
        Description
        - Compute MT fare using discrete travel-time bands.

        Parameters
        - trip_length: trip distance [km].
        
        Output
        - Returns fare for that trip [$].

        Fare policy (time-based, discrete)
        - <= 20 min: 1 band
        - <= 60 min: 2 bands
        - <= 90 min: 3 bands
        - <= 120 min: 4 bands
        - > 120 min: +1 band per additional 30 min started

        Notes
        - `self.fare` is interpreted as the CHF amount per band.
          With `self.fare=2`, this gives 2, 4, 6, 8 CHF for the first 4 bands.
        """
        travel_time_min = self.trip_time(trip_length) * 60.0

        if travel_time_min <= 60.0:
            bands = 2
        elif travel_time_min <= 90.0:
            bands = 3
        elif travel_time_min <= 120.0:
            bands = 4
        else:
            extra_bands = int(np.ceil((travel_time_min - 120.0) / 30.0))
            bands = 4 + extra_bands

        return self.fare * bands

    def trip_time(self, trip_length: float, traveler_type_idx: Optional[int] = None) -> float:
        """
        Description
        - Compute expected in-vehicle travel time.

        Parameters
        - trip_length: trip distance [km].
        
        Output
        - Returns time [hr].
        """
        return self.detour_ratio * trip_length / self.average_speed

    def waiting_time(self, trip_length: float, traveler_type_idx: Optional[int] = None) -> float:
        """
        Description
        - Compute epected waiting time for a certain trip length.

        Parameters
        - trip_length: trip distance [km].

        Output
        - Returns expected waiting time [hr]. 
        """
        return 2 * self.access_time + self.transit_time * (self.n_transfer_per_length * trip_length)


# --------------------------
# MaaS Service
# --------------------------

class MaaS(Service):
    """
    Mobility-as-a-Service (MaaS) platform combining TNC and MT modes.
    """

    def __init__(self,
            ASC: float,
            fare: float,
            share_TNC_per_traveler_type: list[float],
            detour_ratio_TNC: float,
            average_speed_TNC: float,
            capacity_ratio_from_TNC: float | dict[str, float],
            total_service_capacity_TNC: float | dict[str, float],
            average_veh_travel_dist_per_day_TNC: float,
            cost_purchasing_capacity_TNC: float,
            trip_length_per_traveler_type: list[float],
            value_travel_time_per_traveler_type: list[float],
            value_waiting_time_per_traveler_type: list[float],
            detour_ratio_MT: float,
            average_speed_MT: float,
            transit_time_MT: float,
            n_transfer_per_length_MT: float,
            cost_purchasing_capacity_MT: float,
            lambda_M: float,
            logit_scale_mu: float = 1.0,
        ):
        """
        Description  
        - Initialize an integrated Mobility-as-a-Service (MaaS) model combining TNC and MT services.

        Attributes  
        - ASC: alternative specific constant to compute utility in monetary unit [$].  
        - fare: monetary fare per distance [$/km].  
        - share_TNC_per_traveler_type: fraction of MaaS trips served by TNC for each traveler type [-].  

        - detour_ratio_TNC: TNC detour factor applied to trip length [-].  
        - average_speed_TNC: average TNC vehicle speed [km/h].  
        - capacity_ratio_from_TNC: share of TNC capacity reallocated to MaaS service [-].  
        - total_service_capacity_TNC: total TNC service capacity available to MaaS [veh·km].  
        - average_veh_travel_dist_per_day_TNC: average vehicle distance covered per day [km].  
        - cost_purchasing_capacity_TNC: price per veh*km to purchase TNC capacity [$/(veh·km)].  

        - trip_length_per_traveler_type: list of trip lengths per traveler type [km].  
        - value_travel_time_per_traveler_type: list of values of in-vehicle travel time per traveler type [$/hr].  
        - value_waiting_time_per_traveler_type: list of values of waiting time per traveler type [$/hr].  

        - detour_ratio_MT: detour factor in Mass Transit trips [-].  
        - average_speed_MT: average MT transit speed [km/h].  
        - transit_time_MT: average in-vehicle transit time in MT [hr].  
        - n_transfer_per_length_MT: number of transfers per unit trip length in MT [-/km].  
        - cost_purchasing_capacity_MT: price per capacity unit for MT service [$/(veh·km)].  # MM check that units
        - lambda_M: Lagrange multiplier for MaaS capacity constraint [$/(veh·km)]. 

        Attributes (initialized later)   
        - demand_per_traveler_type: dict mapping service name → list of trip counts per traveler type [trips]. # MM : disscuss units [veh] vs [trips] 
                                    Example: {"TNC": [n0, n1, n2], "MT": [..], "MaaS": [...]}  
        - vacant_veh_available: number of vacant vehicles used for MaaS [veh].

        Output  
        - Create a class MaaS and initialize all the attributes.
        """
        super().__init__(name="MaaS")

        self.ASC = ASC
        self.fare = fare
        self.share_TNC_per_traveler_type = np.array(share_TNC_per_traveler_type, dtype=float)
        self.lambda_M = lambda_M
        
        # TNC parameters
        self.detour_ratio_TNC = detour_ratio_TNC
        self.average_speed_TNC = average_speed_TNC
        if isinstance(capacity_ratio_from_TNC, dict):
            self.capacity_ratio_from_TNC_by_name = {
                k: float(v) for k, v in capacity_ratio_from_TNC.items()
            }
        else:
            self.capacity_ratio_from_TNC_by_name = {"TNC": float(capacity_ratio_from_TNC)}

        if isinstance(total_service_capacity_TNC, dict):
            self.total_service_capacity_TNC_by_name = {
                k: float(v) for k, v in total_service_capacity_TNC.items()
            }
        else:
            self.total_service_capacity_TNC_by_name = {"TNC": float(total_service_capacity_TNC)}

        # Backward-compatible aliases for V2-style code paths.
        self.capacity_ratio_from_TNC = float(np.sum(np.array(list(self.capacity_ratio_from_TNC_by_name.values()))))
        self.total_service_capacity_TNC = float(np.sum(np.array(list(self.total_service_capacity_TNC_by_name.values()))))
        self.average_veh_travel_dist_per_day_TNC = average_veh_travel_dist_per_day_TNC
        self.cost_purchasing_capacity_TNC = cost_purchasing_capacity_TNC

        # Traveler informations
        self.trip_length_per_traveler_type: list[float] = trip_length_per_traveler_type
        self.value_travel_time_per_traveler_type: list[float] = value_travel_time_per_traveler_type
        self.value_waiting_time_per_traveler_type: list[float] = value_waiting_time_per_traveler_type
        self.demand_per_traveler_type: dict[str, list[float]] | None = None
        if len(self.share_TNC_per_traveler_type) != len(self.trip_length_per_traveler_type):
            raise ValueError("share_TNC_per_traveler_type must match traveler type count")
        
        # MT parameters
        self.detour_ratio_MT = detour_ratio_MT
        self.average_speed_MT = average_speed_MT
        self.transit_time_MT = transit_time_MT
        self.n_transfer_per_length_MT = n_transfer_per_length_MT
        self.cost_purchasing_capacity_MT = cost_purchasing_capacity_MT

        self.logit_scale_mu: float = logit_scale_mu

    def get_pooled_tnc_capacity(self) -> float:
        pooled = 0.0
        for name, ratio in self.capacity_ratio_from_TNC_by_name.items():
            pooled += ratio * self.total_service_capacity_TNC_by_name.get(name, 0.0)
        return pooled

    def sync_tnc_pool_aliases(self) -> None:
        # Keep legacy scalar fields coherent for old consumers.
        self.capacity_ratio_from_TNC = np.sum(np.array(list(self.capacity_ratio_from_TNC_by_name.values())))
        self.total_service_capacity_TNC = np.sum(np.array(list(self.total_service_capacity_TNC_by_name.values())))

    def trip_fare(self, trip_length: float) -> float:
        """
        Description
        - Compute MaaS fare for a given trip length.

        Parameters
        - trip_length: trip distance [km].
        
        Output
        - Returns fare for that trip [$].
        """
        return self.fare * trip_length

    def _share_for_type(self, traveler_type_idx: Optional[int]) -> float:
        if traveler_type_idx is None:
            if len(self.share_TNC_per_traveler_type) == 1:
                return self.share_TNC_per_traveler_type[0]
            raise ValueError("traveler_type_idx is required when multiple traveler types exist")
        return self.share_TNC_per_traveler_type[traveler_type_idx]

    def trip_time(self, trip_length: float, traveler_type_idx: Optional[int] = None) -> float:
        """
        Description
        - Compute expected travel time for each sub-service (TNC, MT) and then sum both time.

        Parameters
        - trip_length: trip distance [km].

        Internal variables
        - time_TNC: contribution from the portion of the trip performed by TNC [hr].
        - time_MT: contribution from the portion of the trip performed by MT [hr].
        
        Output
        - Returns the summed waiting time from both sub-service (TNC, MT) [hr].
        """
        share_tnc = self._share_for_type(traveler_type_idx)
        time_TNC = self.detour_ratio_TNC * share_tnc * trip_length / self.average_speed_TNC
        time_MT = self.detour_ratio_MT * (1 - share_tnc) * trip_length / self.average_speed_MT
        return time_TNC + time_MT

    def waiting_time(self, trip_length: float, traveler_type_idx: Optional[int] = None) -> float:
        """
        Description
        - Compute expected waiting time for each subservice (TNC, MT) and return the sum.

        Parameters
        - trip_length (float): trip distance in km.

        Internal variables
        - A: exogenous matching factor, A = 2.5 (Zhou et al. (2022), Competition and third-party platform integration in ride-sourcing
            markets. Transportation Research Part B: Methodological, 159, 76-103.)
        - sensitivity_param (float): elasticity w.r.t. vacant vehicles; typical value 0.5 for regular e-hailing services without
            passenger competition in the matching process.

        Output
        - Returns expected waiting time (hours): TNC empirical waiting time plus MT waiting component.
        """
        A = 2.5
        sensitivity_param = 0.5
        share_tnc = self._share_for_type(traveler_type_idx)
        vacant_veh_available = self.find_vacant_veh_available()
        vacant_veh_available = np.where(
            vacant_veh_available <= 0,
            1e-6,
            vacant_veh_available
        ) # safer for autograd
        TNC_waiting_time = A * (vacant_veh_available ** (- sensitivity_param))
        MT_waiting_time = self.transit_time_MT * (self.n_transfer_per_length_MT * (1 - share_tnc) * trip_length)
        return TNC_waiting_time + MT_waiting_time

    def find_vacant_veh_available(self) -> float:
        """
        Description
        - Compute number of idle vehicles in the TNC fleet of MaaS.

        Internal variables
        -total_demand: computed as sum over traveler types of (trip_length_i * allocated_travelers_i) [veh·km]
        
        Output
        - Returns the number of vacant vehicles based on the TNC fleet capacity alocated to MaaS [veh].
        """
        total_demand = np.sum(
            self.share_TNC_per_traveler_type
            * np.array(self.trip_length_per_traveler_type)
            * np.array(self.demand_per_traveler_type[self.name])
        )
        pooled_capacity = self.get_pooled_tnc_capacity()
        vacant = np.array((pooled_capacity - total_demand) / self.average_veh_travel_dist_per_day_TNC)
        # Protect against 0 or negative values to prevent 'nan' in gradients
        return np.where(vacant <= 0, 1e-6, vacant)

    def compute_objective_function(self, params: list[float], travelers: List['Travelers'], services: List[Service], service_index_M: int = 2) -> float:
        """
        Description
        - Compute the MaaS objective function (Lagrangian formulation). See calculation details in report.

        Parameters
        - params: list of operational variables [fare, share_TNC_type_0, ..., share_TNC_type_n, lambda_M].
        - travelers: list of Traveler objects in the simulation.
        - services: list of Service objects in the simulation.
        - service_index_M: index of the MaaS column in U (default 2).

        Internal variables
        - U: shape (i_types, m_services) of utility values U_im for each
            traveler type i and service m.
        - l: array of trip lengths per traveler type (l_i) [km].
        - Q: array of total travelers per type aggregated across services (Q_i) [veh].
        - P: softmax probabilities over services from U (shape n_types x n_services).
        - P_iM: column vector of probabilities of choosing MaaS for each traveler type.
        - sum_l_PiM_Qi: sum_i l_i * P_iM * Q_i (demand assigned to MaaS) [veh*km].
        - sum_PiM_Qi: sum_i P_iM * Q_i (number of travelers choosing MaaS weighted by Q) [veh].

        Objective terms (economic interpretation):
        - term1: fare revenue from MaaS users (negative sign as cost/revenue term) [$].
        - term2: MT capacity purchasing cost for MaaS users [$].
        - term3: TNC capacity purchasing cost associated with MaaS [$]. (EXCLUDED)
        - term4: Lagrangian penalty enforcing capacity constraints (lambda_M) [$].

        Output
        - Returns a scalar value of the objective [$].
        """
        n_types = len(self.trip_length_per_traveler_type)
        # store originals
        fare0 = self.fare
        share_TNC0 = np.array(self.share_TNC_per_traveler_type)
        lambda_M0 = self.lambda_M

        # compute objective with new params
        self.fare = params[0]
        self.share_TNC_per_traveler_type = np.array(params[1:1 + n_types])
        self.lambda_M = params[1 + n_types]
        U = compute_utility_matrix(travelers, services)
        l = np.asarray(self.trip_length_per_traveler_type)
        Q = np.sum(list(self.demand_per_traveler_type.values()), axis=0)
        P = _logit_probabilities(U, self.logit_scale_mu)
        P_iM = P[:, service_index_M]
        sum_l_PiM_Qi = np.sum(l * P_iM * Q)
        term1 = -self.fare * sum_l_PiM_Qi
        term2 = self.cost_purchasing_capacity_MT * np.sum(
            (1 - self.share_TNC_per_traveler_type) * (P_iM * Q)
        )
        term4 = self.lambda_M * (
            np.sum(self.share_TNC_per_traveler_type * l * P_iM * Q)
            - self.get_pooled_tnc_capacity()
        )

        # restore originals
        self.fare = fare0
        self.share_TNC_per_traveler_type = share_TNC0
        self.lambda_M = lambda_M0

        return term1 + term2  + term4 

    def gradient_objective(
            self,
            U: np.ndarray,
            service_index_M: int = 2
        ) -> np.ndarray:
        """
        Description
                - Compute gradient of MaaS objective w.r.t. operational variables
                    (self.fare, self.share_TNC_per_traveler_type and self.lambda_M).

        Parameters
        - U: shape (n_types, n_services) of utility values U_im for each
            traveler type i and service m (monetary units / utility scale).
        - service_index_M: index of the MaaS column in U (default 2).

        Internal variables
        - l: array of trip lengths per traveler type (l_i) [km].
        - Q: array of total travelers per type aggregated across services (Q_i) [veh].
        - P: softmax probabilities over services from U (shape n_types x n_services).
        - P_iM: column vector of probabilities of choosing MaaS for each traveler type.
        - dUdf: partial derivative of utility w.r.t. fare (vector).
        - dUdalph: partial derivative of utility w.r.t. each share_TNC_i (vector).
        - sum_l_PiM_Qi: sum_i l_i * P_iM * Q_i (demand assigned to MaaS) [veh*km].
        - sum_PiM_Qi: sum_i P_iM * Q_i (number of travelers choosing MaaS weighted by Q) [veh].
        - A: exogenous matching factor, A = 2.5 (Zhou et al. (2022), Competition and third-party platform integration in ride-sourcing
            markets. Transportation Research Part B: Methodological, 159, 76-103.)
        - s: sensitivity parameter (elasticity w.r.t. vacant vehicles); typical value 0.5 for regular e-hailing services without
            passenger competition in the matching process.

        Output
        - Returns the gradient vector [dObj/d_fare, dObj/d_share_TNC_0, ..., dObj/d_share_TNC_n, dObj/d_lambda_M].
        """
        # Traveler-specific data
        mu = self.logit_scale_mu
        l = np.asarray(self.trip_length_per_traveler_type)   # l_i
        Q = np.sum(list(self.demand_per_traveler_type.values()), axis=0)  # Q_i
        share = np.asarray(self.share_TNC_per_traveler_type)
        demand_maas = np.asarray(self.demand_per_traveler_type[self.name])
        value_time = np.asarray(self.value_travel_time_per_traveler_type)
        value_wait = np.asarray(self.value_waiting_time_per_traveler_type)

        # Choice probabilities
        P = _logit_probabilities(U, mu)
        P_iM = P[:, service_index_M]

        lQ = l * Q
        sum_l_PiM_Qi = np.sum(lQ * P_iM)

        # ========= PARTIAL DERIVATIVES OF UTILITY ==========
        dUdf = -l
        A, s = 2.5, 0.5
        vacant = self.find_vacant_veh_available()

        # dU_i / d alpha_j matrix captures cross-type coupling through pooled waiting time.
        waiting_prefactor = A * s * (vacant ** (-(s + 1))) / self.average_veh_travel_dist_per_day_TNC
        cross_wait = np.outer(-value_wait * waiting_prefactor, l * demand_maas)

        delta_time = (
            self.detour_ratio_TNC / self.average_speed_TNC * l
            - self.detour_ratio_MT / self.average_speed_MT * l
        )
        diag_extra = (
            -value_time * delta_time
            + value_wait * self.transit_time_MT * self.n_transfer_per_length_MT * l
        )

        dU_dalpha = cross_wait + np.diag(diag_extra)
        # dP_jM/dalpha_i carries an explicit factor mu (logit scale).
        dP_dalpha = mu * (P_iM * (1 - P_iM))[:, None] * dU_dalpha

        # ========= GRAD w.r.t. f_M ==========
        # Each chain-rule sum picks up the same factor mu.
        grad_fM = (
            - sum_l_PiM_Qi
            - self.fare * mu * np.sum(lQ * P_iM * (1 - P_iM) * dUdf)
            + self.cost_purchasing_capacity_MT * mu * np.sum((1 - share) * Q * P_iM * (1 - P_iM) * dUdf)
            + self.lambda_M * mu * np.sum(share * lQ * P_iM * (1 - P_iM) * dUdf)
        )

        # ========= GRAD w.r.t. share_TNC_i ==========
        grad_alpha = (
            - self.fare * (lQ @ dP_dalpha)
            + self.cost_purchasing_capacity_MT * (
                -Q * P_iM + ((1 - share) * Q) @ dP_dalpha
            )
            + self.lambda_M * (
                lQ * P_iM + (share * lQ) @ dP_dalpha
            )
        )

        # ========= GRAD w.r.t. λ_M ==========
        grad_lambdaM = (
            np.sum(share * l * P_iM * Q)
            - self.get_pooled_tnc_capacity()
        )

        return np.concatenate(([grad_fM], np.atleast_1d(grad_alpha), [grad_lambdaM]))

# --------------------------
# Group of Travelers
# --------------------------
class Travelers: 
    '''
    Represents a group of travelers characterized by their size, trip attributes,
    and sensitivity to travel time and waiting time.
    '''

    def __init__(self, number_traveler: int, trip_length: float,
                 value_time: float, value_wait: float,
                 logit_scale_mu: float = 1.0):
        """
        Description
        - Initialize a group of traveler class.

        Attributes
        - number_traveler: number of travelers in this group [trip]. # MM : disscuss the unit
        - trip_length: representative trip distance [km].
        - value_time: monetary value of in-vehicle time [$/hr].
        - value_wait: monetary value of waiting time [$/hr].
        - logit_scale_mu: scale (inverse temperature) of the choice model [-]. mu < 1 softens choices.

        Attributes (initialized later)
        - utilities: smoothed (across simulation day) utility functions per service [$].
        - travelers_per_service: traveler counts per service [trip]. # MM : disscuss the unit

        Output
        - Create a class Travelers and initialize all its attributes
        """
        self.number_traveler = number_traveler
        self.trip_length = trip_length
        self.value_time = value_time
        self.value_wait = value_wait
        self.logit_scale_mu = logit_scale_mu
        self.utilities: Optional[List[float]] = None
        self.travelers_per_service: Optional[List[float]] = None

    def compute_utilities(self, services: list[Service], traveler_type_idx: Optional[int] = None) -> None:
        """
        Description
        - Compute (and smooth) utilities of available services for this group.

        Parameters
        - services: list of class Service available to choose from.

        Side effects
        - sets/updates `self.utilities`, a list with one utility per service.

        Internal variables
        - idx: index of the current service in `services`.
        - U: utility computed for this traveler group for each serivce available [$].

        Output
        - Update self.utilities with new smoothed utilities.
        """
        if not self.utilities:
            self.utilities = [0] * len(services)

        for idx, service in enumerate(services):
            U = service.compute_utility(
                trip_length=self.trip_length,
                value_time=self.value_time,
                value_wait=self.value_wait,
                traveler_type_idx=traveler_type_idx,
            )
            self.utilities[idx] = U # 0.5 * (self.utilities[idx] + U)  # smoothing
        return
    
    def choose_service(self, services: list[Service], traveler_type_idx: Optional[int] = None) -> None:
        '''
        Description
        - Choose service distribution based on a logit model of choice probabilities.
        
        Parameters
        - services: all the class of Service available.

        Internal variables
        - probabilities: probability of choosing a service based on the utilites

        Output
        - Update self.travelers_per_service with the new simulated values. 
        '''
        self.compute_utilities(services, traveler_type_idx=traveler_type_idx)
        utilities = self.logit_scale_mu * np.array(self.utilities)
        utilities -= np.max(utilities) # Stabilize
        exp_utilities = np.exp(utilities)
        probabilities = exp_utilities / np.sum(exp_utilities)
        self.travelers_per_service = probabilities * self.number_traveler
        return

# --------------------------
# Functions
# --------------------------

# ---- Logit helpers ----

def _logit_probabilities(U, mu: float):
    """Softmax with scale (inverse temperature) mu, row-wise on the last axis.

    P_im = exp(mu * U_im) / sum_n exp(mu * U_in). mu < 1 softens choices,
    mu > 1 sharpens them; mu = 1 recovers the standard multinomial logit.
    Used internally by the entity classes' ``compute_objective_function`` and
    ``gradient_objective`` methods.
    """
    scaled = mu * U
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    P = np.exp(scaled)
    return P / np.sum(P, axis=-1, keepdims=True)


def compute_choice_probabilities(utilities: np.ndarray, mu: float = 1.0) -> np.ndarray:
    """
    Description
    - Compute the row-wise stabilised softmax with logit scale ``mu``:
      ``P[i, m] = exp(mu * U[i, m]) / sum_n exp(mu * U[i, n])``.

    Parameters
    - utilities: 2-D array of utilities, shape ``(n_classes, n_services)``.
    - mu: logit scale parameter (``mu < 1`` softens choices).

    Output
    - 2-D array of choice probabilities, same shape as ``utilities``.
    """
    scaled = mu * utilities
    stabilized = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_utilities = np.exp(stabilized)
    return exp_utilities / np.sum(exp_utilities, axis=1, keepdims=True)


# ---- Utility-matrix builders ----

def compute_utilities(travelers: list['Travelers'], services: list[Service]) -> np.ndarray:
    """
    Description
    - Compute the utility matrix U for all traveler classes and services.

    Parameters
    - travelers: list of Travelers objects.
    - services: list of Service objects.

    Output
    - 2-D array U of shape (n_classes, n_services); ``U[i, m]`` is the utility
      of service ``m`` for class ``i``.
    """
    utilities = []
    for traveler_idx, traveler in enumerate(travelers):
        row = []
        for service in services:
            utility = service.compute_utility(
                traveler.trip_length,
                traveler.value_time,
                traveler.value_wait,
                traveler_type_idx=traveler_idx,
            )
            row.append(utility)
        utilities.append(row)
    return np.array(utilities)


def compute_utility_matrix(travelers: List['Travelers'], services: List[Service]) -> np.ndarray:
    """
    Description
    - Autograd-compatible variant of ``compute_utilities``. Used inside
      ``compute_objective_function`` where the utility-matrix construction must
      flow through autograd's ArrayBox machinery (extra ``np.sum`` / ``np.squeeze``
      coerce per-service utilities to plain scalars autograd can chain through).

    Parameters
    - travelers: list of Traveler objects in the simulation.
    - services: list of Service objects in the simulation.

    Output
    - Returns U: shape (n_types, n_services) of utility values U_im for each
      traveler type i and service m (monetary units / utility scale).
    """
    rows = []
    for i, traveler in enumerate(travelers):
        row = []
        for m, service in enumerate(services):
            val = service.compute_utility(
                trip_length = traveler.trip_length,
                value_time   = traveler.value_time,
                value_wait   = traveler.value_wait,
                traveler_type_idx = i,
            )
            # Aggregate to scalar if compute_utility returns a vector
            utility = np.sum(val)            # or other aggregator: dot(weights, val), etc.
            utility = np.squeeze(utility)    # ensure shape == ()
            row.append(utility)
        rows.append(row)

    # Build an autograd numpy array from Python lists of ArrayBox scalars
    U = np.array(rows, dtype=float)   # autograd.numpy.array
    return U


# ---- Dispatch ----

def distribute_travelers(travelers: list['Travelers'], services: list[Service]) -> dict[str, list[float]]:
    """
    Description
    - Allocate groups of travelers among services using each group's choice model.

    Parameters
    - travelers: list of traveler groups.
    - services: list of Service instances available.

    Output
    - Returns allocation: mapping service name -> list of floats
                          giving the number of travelers of each traveler group assigned to that service.
                          Example structure: allocation = {"TNC": [n_type0, n_type1, n_type2],
                                                           "MT":  [..],
                                                           "MaaS":[..]}
    """
    allocation = {service.name: [0] * len(travelers) for service in services}
    for type_i, traveler in enumerate(travelers):
        traveler.choose_service(services, traveler_type_idx=type_i)
        for index, service in enumerate(services):
            allocation[service.name][type_i] += traveler.travelers_per_service[index]
    return allocation


# ---- Constraint projection on upper-level parameter vectors ----

def project_tnc_params(params):
    """Project the TNC parameter vector onto its feasible set:
    ``fare >= 0``, ``cap_ratio in [0, 1]``, ``lambda_T >= 0``."""
    fare, cap_ratio, lambda_t = params
    fare = max(fare, 0.0)
    cap_ratio = np.clip(cap_ratio, 0, 1.0)
    lambda_t = max(lambda_t, 0.0)
    return np.array([fare, cap_ratio, lambda_t])


def project_maas_params(params, n_types):
    """Project the MaaS parameter vector ``[f_M, alpha_1, ..., alpha_K, lambda_M]``
    onto its feasible set: ``f_M >= 0``, every ``alpha_i in [0, 1]``, ``lambda_M >= 0``."""
    fare = params[0]
    share_tnc_per_type = params[1:1 + n_types]
    lambda_m = params[1 + n_types]
    fare = max(fare, 0.0)
    share_tnc_per_type = np.clip(share_tnc_per_type, 0.0, 1.0)
    lambda_m = max(lambda_m, 0.0)
    return np.concatenate(([fare], share_tnc_per_type, [lambda_m]))


# ---- Simulation-loop bookkeeping ----

def store_allocations(day, travelers, services, allocation, allocation_history, allocation_by_type):
    """
    Description
    - Append the current day's allocation to the per-service totals history and
      to the per-class history; the slot for ``day`` is overwritten if already
      present (so callers can re-store after smoothing).

    Parameters
    - day: zero-indexed current day.
    - travelers: list of Travelers objects.
    - services: list of Service objects.
    - allocation: dict mapping service name to per-class allocation list.
    - allocation_history: dict mapping service name to a list of daily totals.
    - allocation_by_type: dict mapping service name to a list (length == K) of
      per-day allocation lists, one per traveler class.

    Output
    - The two updated history dicts.
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