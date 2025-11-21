"""
Inflow generator based on: https://github.com/gduthe/windfarm-gnn/blob/main/graph_farms/inflow_gen.py from the paper [1], the IEC standard [2] and Dimitrov et al. [3].

[1] - de Santos, F. N., Duthé, G., Abdallah, I., Réthoré, P.-É., Weijtjens, W., Chatzi, E., & Devriendt, C. (2024). Multivariate prediction on wake-affected wind
turbines using graph neural networks. Journal of Physics: Conference Series, 2647(11)(XII International Conference on Structural Dynamics), 112006. https://doi.org/10.3929/ethz-b-000674010
[2] - IEC 61400-1:2019. Wind turbines – Part 1: Design requirements. International Electrotechnical Commission.
[3] - Dimitrov, N., Kelly, M. C., Vignaroli, A., & Berg, J. (2018). From wind to loads: wind turbine site-specific load estimation with surrogate models trained on high-fidelity load databases. Wind Energy Science, 3(2), 767–790. https://doi.org/10.5194/wes-3-767-2018
"""

import numpy as np
from scipy.stats import qmc, weibull_min
import matplotlib.pyplot as plt


def IEC_61400_1_2019_class_interpreter(
    wt_class: str = "I", ti_charataristics: str = "B"
):
    """IEC 61400-1:2019. Ch. 6 Section 2, Table 1"""
    if wt_class == "I":
        V_ave = 10.0
        V_ref = 50.0
    elif wt_class == "II":
        V_ave = 8.5
        V_ref = 42.5
    elif wt_class == "III":
        V_ave = 7.5
        V_ref = 37.5
    V_ref_tropical = 57  # same for all turbine classes

    if ti_charataristics == "A+" or "very high":
        I_ref = 0.18
    elif ti_charataristics == "A" or "high":
        I_ref = 0.16
    elif ti_charataristics == "B" or "medium":
        I_ref = 0.14
    elif ti_charataristics == "C" or "low":
        I_ref = 0.12

    inflow_settings = {
        "V_ave": V_ave,
        "Iref": I_ref,
        "V_ref": V_ref,
        "V_ref_tropical": V_ref_tropical,
    }
    return inflow_settings


class InflowGenerator:
    """Inflow generation class, generates inflow conditions for a wind farm.

    kwargs:
    inflow_settings: dict containing the following keys:
        V_ave: mean wind speed [m/s] (IEC)
        Iref: reference turbulence intensity (IEC)
    turbine_settings: dict containing the following keys:
        cutin_u: cut-in wind speed [m/s]
        cutout_u: cut-out wind speed [m/s]
        height_above_ground: height above ground [m]
    """

    def __init__(self, **kwargs):
        # check that kwargs contain the necessary dicts
        assert {"inflow_settings", "turbine_settings"}.issubset(kwargs.keys())

        # store the configuration settings
        self.inflow_settings = kwargs["inflow_settings"]
        self.turbine_settings = kwargs["turbine_settings"]

        # initialize Sobol sampler for consistent sampling across independent parameters
        self.sampler = qmc.Sobol(d=2, scramble=True)  # Duthe uses Sobol, it is better
        # self.sampler = qmc.Halton(d=2, scramble=True) # Dimitrov uses Halton

    def _gen_wind_velocities(self, random_samples: np.array):
        """Generatae wind speeds based on IEC 61400-1:2019 Ch. eq. 8
        using a Rayleigh distribution for wind speed"""

        # compute scale parameter, sqrt(pi) is an alteration from IEC
        C = 2 * self.inflow_settings["V_ave"] / np.sqrt(np.pi)

        # Scale and shift probabiliteies to match cut-in and cut-out wind speeds
        bounds = np.array(
            [self.turbine_settings["cutin_u"], self.turbine_settings["cutout_u"]]
        )
        P_bounds = weibull_min.cdf(bounds, c=2, loc=0, scale=C)
        x = P_bounds[0] + random_samples * (P_bounds[1] - P_bounds[0])

        # inverse transform to get wind speeds
        u = weibull_min.ppf(x, c=2, loc=0, scale=C)
        return u

    def _gen_turbulence(self, u: np.array, random_samples: np.array, method="Dimitrov"):
        """Generate turbulence intensity based on wind speed"""

        if method == "NTM":
            """Normal Turbulence Model (NTM) IEC 61400-1:2019 Ch. 6.3.2.3 eq. (10)"""
            sigma_1 = self.inflow_settings["Iref"] * (0.75 * u + 5.6)

        elif method == "Dimitrov":
            """Dimitrov et. al. (2018). From wind to load https://doi.org/10.5194/wes-3-767-2018
            Range of sigma_u and uniform sampling based on Table 1."""

            I_refAp = 0.18  # The IEC 61400-1 reference turbulence intensity for A+
            # Upper and lower bounds below from Table 1 in Dimitrov et al. (2018)
            sigma_upper = I_refAp * (6.8 + 0.75 * u + 3 * (10 / u) ** 2)
            sigma_lower = 0.0025 * u
            sigma_ranges = sigma_upper - sigma_lower
            sigma_1 = sigma_lower + sigma_ranges * random_samples  # Uniform samples
        else:
            raise ValueError("Invalid method. Choose either 'NTM' or 'Dimitrov'")

        ti = sigma_1 / u
        return ti

    def generate_inflows(self, num_samples: int, output_type: str = "array"):
        # generates all the boundary conditions using sobol sampling
        samples = self.sampler.random(n=num_samples)
        u = self._gen_wind_velocities(samples[:, 0])
        ti = self._gen_turbulence(u, samples[:, 1])
        if output_type == "dict":
            output = {
                "u": u,
                "ti": ti,
            }
        elif output_type == "array":
            output = np.vstack((u, ti)).T
        return output


if __name__ == "__main__":
    from py_wake.examples.data.dtu10mw import DTU10MW
    from py_wake.examples.data.dtu10mw import power_curve as power_curve_dtu10mw

    wt = DTU10MW()

    hub_height = wt.hub_height()
    cutin = power_curve_dtu10mw[:, 0].min()
    cutout = power_curve_dtu10mw[:, 0].max()

    inflow_settings = IEC_61400_1_2019_class_interpreter(
        wt_class="I", ti_charataristics="B"
    )

    turbine_settings = {
        "cutin_u": cutin,
        "cutout_u": cutout,
        "height_above_ground": wt.hub_height(),
    }

    inflow_gen = InflowGenerator(
        inflow_settings=inflow_settings, turbine_settings=turbine_settings
    )
    num_samples = 1024 * 5
    if 0:  # Some lines for debugging
        samples = inflow_gen.sampler.random_base2(m=int(np.ceil(np.log2(num_samples))))[
            :num_samples
        ]
        u = inflow_gen._gen_wind_velocities(samples[:, 0])
        ti = inflow_gen._gen_turbulence(u, samples[:, 1])

    inflow = inflow_gen.generate_inflows(num_samples)

    plt.hist(inflow["u"], bins=50)
    plt.title("Wind speed distribution", fontsize=14)
    plt.xlabel("Wind speed [m/s]", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

    plt.hist(inflow["ti"], bins=50)
    plt.title("Turbulence intensity distribution", fontsize=14)
    plt.xlabel("Turbulence intensity", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

    plt.scatter(inflow["u"], inflow["ti"])
    plt.title("Wind speed vs turbulence intensity", fontsize=14)
    plt.xlabel("Wind speed [m/s]", fontsize=14)
    plt.ylabel("Turbulence intensity", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
