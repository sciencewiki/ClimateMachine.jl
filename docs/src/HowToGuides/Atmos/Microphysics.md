# Microphysics

The `Microphysics.jl` module describes a 1-moment bulk parameterization of cloud microphysical processes.
The module is based on the ideas of
[Kessler\_1995](https://www.sciencedirect.com/science/article/pii/016980959400090Z),
[Grabowski\_1998](https://journals.ametsoc.org/doi/full/10.1175/1520-0469%281998%29055%3C3283%3ATCRMOL%3E2.0.CO%3B2 )
and [Kaul\_et\_al\_2015](https://journals.ametsoc.org/doi/10.1175/MWR-D-14-00319.1)

The cloud microphysics variables are expressed as specific humidities:
  - `q_tot` - total water specific humidity,
  - `q_vap` - water vapor specific humidity,
  - `q_liq` - cloud water specific humidity,
  - `q_ice` - cloud ice specific humidity,
  - `q_rai` - rain specific humidity,
  - `q_sno` - snow specific humidity.

## Assumed particle size relationships

Particles are assumed to follow mass(radius), cross section(radius), terminal velocity(radius)
relationships defined as power laws.
The coefficients are defined in
[CLIMAParameters.jl](https://github.com/CliMA/CLIMAParameters.jl) package
and are shown in the Table below.
For rain and ice they correspond to spherical liquid water drops and ice particles, respectively.
There is no assumed particle shape for snow, and the relationships are based on empirical values
from [Grabowski\_1998](https://journals.ametsoc.org/doi/full/10.1175/1520-0469%281998%29055%3C3283%3ATCRMOL%3E2.0.CO%3B2 ).

```math
m(r) = \alpha r^{\beta}
```
```math
a(r) = \gamma r^{\delta}
```
```math
v_{term}(r) = \zeta r^{\eta}
```
where:
 - ``r`` is the particle radius,
 - ``\alpha, \, \beta, \, \gamma, \, \delta, \, \zeta, \, \eta \,`` are the coefficients.

These coefficients,
smiliarliy to all other microphysics parameters,
are not hardcoded in the final microphysics parameterizations.
The goal is to allow easy flexibility when calibrating the microphysics parameters.
With that said, the assumption about the shape of the particles is used twice
when deriving the microphysics formulae:
 - The mass transfer equation (\ref{eq:mass_rate}) used in snow autoconversion, rain evaporation,
   snow sublimation and snow melt rates is derived assuming spherical particle shapes.
   To correct for non-spherical shape it should be multiplied by a function of the
   particle aspect ratio.
 - The geometric collision kernel used for deriving rain-snow accretion rate
   assumes that both colliding particles are spherical.
   It does not take into account the reduced cross-section of snow particles that is used
   when modeling snow - cloud water and snow - cloud ice accretion.

|    symbol          |         definition                              | units                    | default value                         |
|--------------------|-------------------------------------------------|--------------------------|---------------------------------------|
|``\alpha^{rai}``    | coefficient in mass(radius) for rain            | ``\frac{kg}{m^3}``       | ``\frac{4}{3} \, \pi \, \rho_{water}``|
|``\beta^{rai}``     | exponent in mass(radius) for rain               | -                        | ``3``                                 |
|``\gamma^{rai}``    | coefficient in cross section(radius) for rain   | -                        | ``\pi``                               |
|``\delta^{rai}``    | exponent in cross section(radius) for rain      | -                        | ``2``                                 |
|``\eta^{rai}``      | exponent in velocity(radius) for rain           | -                        | ``0.5``                               |
|                    |                                                 |                          |                                       |
|``\alpha^{ice}``    | coefficient in mass(radius) for ice             | ``\frac{kg}{m^3}``       | ``\frac{4}{3} \, \pi \, \rho_{ice}``  |
|``\beta^{ice}``     | exponent in mass(radius) for ice                | -                        | ``3``                                 |
|                    |                                                 |                          |                                       |
|``\alpha^{sno}``    | coefficient in mass(radius) for snow            | ``\frac{kg}{m^2}``       | ``0.1``                               |
|``\beta^{sno}``     | exponent in mass(radius) for snow               | -                        | ``2``                                 |
|``\gamma^{sno}``    | coefficient in cross section(radius) for snow   | -                        | ``0.3 \, \pi``                        |
|``\delta^{sno}``    | exponent in cross section(radius) for snow      | -                        | ``2``                                 |
|``\zeta^{sno}``     | coefficient in velocity(radius) for snow        | ``\frac{m^{3/4}}{s}``    | ``2^{9/4}``                           |
|``\eta^{sno}``      | exponent in velocity(radius) for snow           | -                        | ``0.25``                              |

where:
 - ``\rho_{water}`` is the density of water,
 - ``\rho_{ice}`` is the density of ice.

``\gamma^{sno}`` corresponds to parameter ``\alpha`` in eq(16b) in
from [Grabowski\_1998](https://journals.ametsoc.org/doi/full/10.1175/1520-0469%281998%29055%3C3283%3ATCRMOL%3E2.0.CO%3B2 ).

The terminal velocity of an individual rain drop is defined by the balance between the gravitational acceleration
(taking into account the density difference between water and air) and the drag force.
Therefore the ``\zeta^{rai}`` is defined as
```math
\begin{equation}
\zeta^{rai} = \left(\frac{8}{3 \, C_{drag}} \left( \frac{\rho_{water}}{\rho} -1 \right) \right)^{1/2} (g)^{1/2}
\label{eq:vdrop}
\end{equation}
```
where:
 - ``g`` is the gravitational acceleration,
 - ``C_{drag}`` is the drag coefficient,
 - ``\rho`` is the density of air.\

!!! note

    It would be great to replace the above simple power laws with more accurate relationships.
    For example:
    [Khvorostyanov\_and\_Curry\_2002](https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%282002%29059%3C1872%3ATVODAC%3E2.0.CO%3B2)

## Assumed particle size distributions

The particle size distributions are assumed to follow Marshall-Palmer distribution
[Marshall\_and\_Palmer\_1948](https://journals.ametsoc.org/doi/abs/10.1175/1520-0469%281948%29005%3C0165%3ATDORWS%3E2.0.CO%3B2)
 eq. 1:
```math
\begin{equation}
n(r) = n_{0} exp\left(- \lambda \, r \right)
\end{equation}
```
where:
 - ``n_{0}`` and ``\lambda`` are the Marshall-Palmer distribution parameters.

The ``n_0`` for rain and ice is assumed constant.
The ``n_0`` for snow is defined as
```math
\begin{equation}
n_0^{sno} = \mu^{sno} (\rho q_{sno})^{\nu^{sno}}
\end{equation}
```
where:
 - ``\mu^{sno}`` and ``\nu^{sno}`` are the coefficients

The coefficients are defined in `MicrophysicsParameters` module and are shown in the Table below.

|    symbol       |         definition                             | units              | default value       |
|-----------------|------------------------------------------------|--------------------|---------------------|
|``n_{0}^{rai}``  | rain drop size distribution parameter          | ``\frac{1}{m^4}``  | ``16 \cdot 10^6``   |
|``n_{0}^{ice}``  | cloud ice size distribution parameter          | ``\frac{1}{m^4}``  | ``2 \cdot 10^7``    |
|``\mu^{sno}``    | snow size distribution parameter coefficient   | ``?``              | ``4.36 \cdot 10^9`` |
|``\nu^{sno}``    | snow size distribution parameter exponent      | ``?``              | ``0.63``            |

``n_{0}^{rai}`` is based on the original
[Marshall\_and\_Palmer\_1948](https://journals.ametsoc.org/doi/abs/10.1175/1520-0469%281948%29005%3C0165%3ATDORWS%3E2.0.CO%3B2)
paper. The rest of the parameters is taken from
[Kaul\_et\_al\_2015](https://journals.ametsoc.org/doi/10.1175/MWR-D-14-00319.1)

The ``\lambda`` parameter is defined as
```math
\begin{equation}
\lambda = \left(\frac{\alpha \, n_0 \, \Gamma(\beta + 1)}{\rho \, q}\right)^{\frac{1}{\beta + 1}}
\end{equation}
```
where:
 - ``q`` is rain, snow or ice specific humidity
 - ``\alpha``, ``\beta`` and ``n_0`` are the corresponding mass-radius and size distribution parameters\
 - ``\Gamma()`` is the gamma function

The cloud-ice size distribution is only used when computing snow autoconversion rate.
In the derivation of different accretion rates the cloud ice, similar to cloud water,
is treated as continuous.

!!! note
     - Do we want to keep the ``n_0`` for rain constant and ``n_0`` for snow empirical?

     - Do we want to test different size distributions?

## Parameterized processes

Parameterized processes include:
  - diffusion of water vapour on cloud droplets and cloud ice crystals
    modeled as a relaxation to equilibrium
  - autoconversion of rain and snow,
  - accretion,
  - evaporation of rain water, sublimation and melting of snow
  - sedimentation of rain and snow with mass weighted average terminal velocity
    (Water vapour, cloud water and cloud ice are part of the working fluid and do not sediment.)

Parameters used in the parameterization are defined in `MicrophysicsParameters` module. They consist of:

|    symbol                  |         definition                                        | units                    | default value         |
|----------------------------|-----------------------------------------------------------|--------------------------|-----------------------|
|``C_{drag}``                | rain drop drag coefficient                                | -                        | ``0.55``              |
|``\tau_{cond\_evap}``       | cloud water condensation/evaporation timescale            | ``s``                    | ``10``                |
|``\tau_{sub\_resub}``       | cloud ice condensation/evaporation timescale              | ``s``                    | ``10``                |
|``\tau_{acnv}``             | cloud to rain water autoconversion timescale              | ``s``                    | ``10^3``              |
|``q_{liq\_threshold}``      | cloud to rain water autoconversion threshold              | -                        | ``5 \cdot 10^{-4}``   |
|``r_{is}``                  | threshold particle radius between ice and snow            | ``m``                    | ``62.5 \cdot 10^{-6}``|
|``E_{lr}``                  | collision efficiency between rain drops and cloud droplets| -                        | ``0.8``               |
|``E_{ls}``                  | collision efficiency between snow and cloud droplets      | -                        | ``0.1``               |
|``E_{ir}``                  | collision efficiency between rain drops and cloud ice     | -                        | ``1``                  |
|``E_{is}``                  | collision efficiency between snow and cloud ice           | -                        | ``0.1``                |
|``E_{rs}``                  | collision efficiency between rain drops and snow          | -                        | ``1``                  |
|``a_{vent}^{rai}, b_{vent}^{rai}``      | rain drop ventilation factor coefficients     | -                        | ``1.5 \;``,``\; 0.53`` |
|``a_{vent}^{sno}, b_{vent}^{sno}``      | snow ventilation factor coefficients          | -                        | ``0.65 \;``,``\; 0.44``|
|``K_{therm}``               | thermal conductivity of air                               | ``\frac{J}{m \; s \; K}``| ``2.4 \cdot 10^{-2}`` |
|``\nu_{air}``               | kinematic viscosity of air                                | ``\frac{m^2}{s}``        | ``1.6 \cdot 10^{-5}`` |
|``D_{vapor}``               | diffusivity of water vapor                                | ``\frac{m^2}{s}``        | ``2.26 \cdot 10^{-5}``|


The default value of ``C_{drag}`` for rain is chosen such that the resulting mass averaged terminal velocity is close to
the empirical terminal velocity formulation in
[Smolarkiewicz\_and\_Grabowski\_1996](https://journals.ametsoc.org/doi/abs/10.1175/1520-0493%281996%29124%3C0487%3ATTLSLM%3E2.0.CO%3B2).
The default values of ``a_{vent}^{rai}, b_{vent}^{rai}`` are chosen so that at
``q_{tot} = 15 g/kg`` and ``T=288K`` the resulting rain evaporation rate is close to the empirical
rain evaporation rate from
[Smolarkiewicz\_and\_Grabowski\_1996](https://journals.ametsoc.org/doi/abs/10.1175/1520-0493%281996%29124%3C0487%3ATTLSLM%3E2.0.CO%3B2).
The default values of ``a_{vent}^{sno}, b_{vent}^{sno}`` are from
[Kaul\_et\_al\_2015](https://journals.ametsoc.org/doi/10.1175/MWR-D-14-00319.1).
The default values of ``\tau_{acnv}`` and ``q_{liq\_threshold}`` are based on
[Smolarkiewicz\_and\_Grabowski\_1996](https://journals.ametsoc.org/doi/abs/10.1175/1520-0493%281996%29124%3C0487%3ATTLSLM%3E2.0.CO%3B2).
The ``r_{is}`` threshold is taken from
[Harrington_1995](https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%281995%29052%3C4344%3APOICCP%3E2.0.CO%3B2).
The ``E_{lr}`` is taken from eq(16a) in
[Grabowski\_1998](https://journals.ametsoc.org/doi/full/10.1175/1520-0469%281998%29055%3C3283%3ATCRMOL%3E2.0.CO%3B2 ).
The ``E_{is}`` and ``E_{rs}`` are taken from
[Morrison\_and\_Gettelman\_2008](https://journals.ametsoc.org/doi/10.1175/2008JCLI2105.1)
The ``E_{ir}`` is taken from
[Rutledge\_and\_Hobbs\_1984](https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%281984%29041%3C2949%3ATMAMSA%3E2.0.CO%3B2).
The ``E_{ls}`` is taken from
[Rutledge\_and\_Hobbs\_1983](https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%281983%29040%3C1185%3ATMAMSA%3E2.0.CO%3B2).

### Ventilation factor

The ventilation factor parametrizes the increase in the mass and heat exchange for falling particles.
Following [Seifert\_and\_Beheng\_2006](https://link.springer.com/article/10.1007/s00703-005-0112-4) eq. 24
the ventilation factor is defined as:
```math
\begin{equation}
F(r) = a_{vent} + b_{vent}  N_{Sc}^{1/3} N_{Re}(r)^{1/2}
\label{eq:ventil_factor}
\end{equation}
```
where:
 - ``a_{vent}``, ``b_{vent}`` are coefficients,
 - ``N_{Sc}`` is the Schmidt number,
 - ``N_{Re}`` is the Reynolds number of a falling particle.

The Schmidt number is assumed constant:
```math
N_{Sc} = \frac{\nu_{air}}{D_{vapor}}
```
where:
 - ``\nu_{air}`` is the kinematic viscosity of air,
 - ``D_{vapor}`` is the diffusivity of water.

The Reynolds number of a spherical drop is defined as:
```math
N_{Re} = \frac{2 \, r \, v_{term}(r)}{\nu_{air}}
```
Applying the terminal velocity(radius) relationship results in
```math
\begin{equation}
F(r) = a_{vent} +
       b_{vent} \, \left(\frac{\nu_{air}}{D_{vapor}}\right)^{\frac{1}{3}} \,
       \left(\frac{2\zeta}{\nu_{air}}\right)^{\frac{1}{2}} r^{\frac{\eta + 1}{2}}
\label{eq:vent_factor}
\end{equation}
```

### Terminal velocity

The mass weighted terminal velocity ``v_t`` is defined following
[Ogura\_and\_Takahashi\_1971](https://journals.ametsoc.org/doi/abs/10.1175/1520-0493%281971%29099%3C0895%3ANSOTLC%3E2.3.CO%3B2):
```math
\begin{equation}
v_t = \frac{\int_0^\infty n(r) \, m(r) \, v_{term}(r) \, dr}{\int_0^\infty n(r) \, m(r) \, dr}
\label{eq:vt}
\end{equation}
```
Integrating over the assumed Marshall-Palmer distribution and using the mass-radius and velocity-radius relationships
results in
```math
\begin{equation}
v_t = \zeta \lambda^{-\eta} \frac{\Gamma(\eta + \beta + 1)}{\Gamma(\beta + 1)}
\end{equation}
```

!!! note

    Assuming a constant drag coefficient is an approximation and it should be size and flow dependent,
    see [drag_coefficient](https://www.grc.nasa.gov/www/K-12/airplane/dragsphere.html).
    In general we should implement these terminal velocity parameterizations:
    [Khvorostyanov\_and\_Curry\_2002](https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%282002%29059%3C1872%3ATVODAC%3E2.0.CO%3B2)

### Cloud water condensation/evaporation

Condensation and evaporation of cloud water is parameterized as a relaxation to equilibrium value at the current time step.
```math
\begin{equation}
  \left. \frac{d \, q_{liq}}{dt} \right|_{cond, evap} = \frac{q^{eq}_{liq} - q_{liq}}{\tau_{cond\_evap}}
\end{equation}
```
where:
 - ``q^{eq}_{liq}`` - liquid water specific humidity in equilibrium,
 - ``q_{liq}`` - liquid water specific humidity,
 - ``\tau_{cond\_evap}`` - relaxation timescale (parameter in `MicrophysicsParameters` module).

### Cloud ice sublimation/resublimation

Sublimation and resublimation of cloud ice is parameterized as a relaxation to equilibrium value at the current time step.
```math
\begin{equation}
  \left. \frac{d \, q_{ice}}{dt} \right|_{sub, resub} = \frac{q^{eq}_{ice} - q_{ice}}{\tau_{sub\_resub}}
\end{equation}
```
where:
 - ``q^{eq}_{ice}`` - ice specific humidity in equilibrium,
 - ``q_{ice}`` - ice specific humidity,
 - ``\tau_{sub\_resub}`` - relaxation timescale (parameter in `MicrophysicsParameters` module).

!!! note
    Both ``\tau_{cond\_evap}`` and ``\tau_{sub\_resub}`` are assumed constant here.
    It would be great to make the relaxation time a function of available condensation nuclei, turbulence intensity, etc.
    See works by [prof Raymond Shaw](https://www.mtu.edu/physics/department/faculty/shaw/) for hints.

### Rain autoconversion

Rain autoconversion defines the rate of conversion form cloud to rain water due to
collisions between cloud droplets. It is parameterized following
[Kessler\_1995](https://www.sciencedirect.com/science/article/pii/016980959400090Z):

```math
\begin{equation}
  \left. \frac{d \, q_{rai}}{dt} \right|_{acnv} = \frac{max(0, q_{liq} - q_{liq\_threshold})}{\tau_{acnv}}
\end{equation}
```
where:
 - ``q_{liq}`` - liquid water specific humidity,
 - ``\tau_{acnv}`` - timescale (parameter in `MicrophysicsParameters` module),
 - ``q_{liq\_threshold}`` - autoconversion (parameter in `MicrophysicsParameters` module).

!!! note
    This is the simplest possible autoconversion parameterization.
    It would be great to implement others and test the impact on precipitation.
    See for example [Wood\_2005](https://journals.ametsoc.org/doi/full/10.1175/JAS3530.1)
    Table 1 for other simple choices.

### Snow autoconversion

Snow autoconversion defines the rate of conversion form cloud ice to snow due
the growth of cloud ice by sublimation (deposition of water vapor).
It is defined as the change of mass of cloud ice for cloud ice particles
larger than threshold ``r_{is}``.
See [Harrington_1995](https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%281995%29052%3C4344%3APOICCP%3E2.0.CO%3B2)
for derivation.

```math
\begin{equation}
  \left. \frac{d \, q_{sno}}{dt} \right|_{acnv} =
  \frac{1}{\rho} \frac{d}{dt} \left( \int_{r_{is}}^{\infty} m(r) n(r) dr \right) =
  \left. \frac{1}{\rho} \frac{dr}{dt} \right|_{r=r_{is}} m(r_{is}) n(r_{is}) + \frac{1}{\rho} \int_{r_{is}}^{\infty} \frac{dm}{dt} n(r) dr
\end{equation}
```
The ``\frac{dm}{dt}`` is obtained by solving the water vapor diffusion equation in spherical coordinates
and linking the changes in temperature at the drop surface to the changes in saturated vapor pressure via the Clausius-
Clapeyron equation, following
[Mason\_1971](https://global.oup.com/academic/product/the-physics-of-clouds-9780199588046?cc=us&lang=en&).

For the simplest case of spherical particles and not taking into account ventilation effects:
```math
\begin{equation}
\frac{dm}{dt} = 4 \pi \, r \, (S - 1) \, G(T, p)
\label{eq:mass_rate}
\end{equation}
```
where:
 - ``S(q_{vap}, q_{vap}^{sat}) = \frac{q_{vap}}{q_{vap}^{sat}} - 1`` is commonly labeled as supersaturation,
 - ``q_{vap}^{sat}`` is the saturation vapor specific humidity,
 - ``G(T) = \left(\frac{L_s}{KT} \left(\frac{L_s}{R_v T} - 1 \right) + \frac{R_v T}{p_{vap}^{sat} D} \right)^{-1}``
     combines the effects of thermal conductivity and water diffusivity.
 - ``L_s`` is the latent heat of sublimation,
 - ``K_{thermo}`` is the thermal conductivity of air,
 - ``R_v`` is the gas constant of water vapor,
 - ``D_{vapor}`` is the diffusivity of water vapor

Using eq. (\ref{eq:mass_rate}) and the assumed mass-radius relationship we obtain
```math
\begin{equation}
\frac{dr}{dt} = \frac{4 \pi \, (S - 1) \, G(T, p)}{\alpha^{ice} \beta^{ice}} r^{2-\beta^{ice}}
\label{eq:r_rate}
\end{equation}
```
Finally the snow autoconversion rate is computed as
```math
\begin{equation}
  \left. \frac{d \, q_{sno}}{dt} \right|_{acnv} =
   \frac{1}{\rho} 4 \pi \, (S-1) \, G(T,p) \, n_0^{ice} \, exp(-\lambda_{ice} r_{is})
   \left( \frac{r_{is}^2}{\beta^{ice}} + \frac{r_{is} \lambda_{ice} +1}{\lambda_{ice}^2} \right)
\end{equation}
```
!!! note
    We should include ventilation effects.

    For non-spherical particles the mass rate of growth
    should be multiplied by a function depending on the particle aspect ratio.
    For functions proposed for different crystal habitats see
    [Harrington_1995](https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%281995%29052%3C4344%3APOICCP%3E2.0.CO%3B2) Appendix B.

### Accretion

Accretion defines the rates of conversion between different categories due to collisions between particles.

For the case of collisions bewteen suspended water (cloud water or cloud ice)
and falling water (rain or snow) the sink of suspended water is defined as:

```math
\begin{equation}
\left. \frac{d \, q_{s}}{dt} \right|_{accr} =  - \int_0^\infty n_f(r) \, a_f(r) \, v_{term}(r) E_{sf} q_{s} dr
\label{eq:accr_1}
\end{equation}
```
where:
 - ``s`` subscript indicates suspended water category (cloud water or cloud ice)
 - ``f`` subscript indicates falling water category (rain or snow)
 - ``E_{sf}`` is the collision efficiency.

Integrating over the distribution yeilds:
```math
\begin{equation}
\left. \frac{d \, q_s}{dt} \right|_{accr} =
  - n_{0f} \, \zeta_f \, \gamma_f \, q_s \, E_{sf} \, \frac{\Gamma(\eta_f + \delta_f + 1)}{\lambda_f^{\eta_f + \delta_f +1}}
\label{eq:accrfin}
\end{equation}
```
For the case of cloud water/rain and cloud ice/snow collisions, the sink of suspended water
becomes simply the source for falling water.
For the case of cloud water and snow collisions for temperatures below freezing,
the sink of cloud water is a source for snow.
For temperatures above freezing, the accreted cloud droplets along with some melted snow
are converted to rain.
In this case eq. (\ref{eq:accrfin}) describes the sink of cloud water.
The sink of snow is proportional to the sink of cloud water with the coefficient ``\frac{c_w}{L_f}(T - T_{freeze})``.

The collisions between cloud ice and rain create snow.
The source of snow in this case is a sum of sinks from cloud ice and rain.
The sink of cloud ice is defined by eq. (\ref{eq:accrfin}).
The sink of rain is defined as:

```math
\begin{equation}
\left. \frac{d \, q_{rai}}{dt} \right|_{accr\_ri} =
  - \int_0^\infty \int_0^\infty
  \frac{1}{\rho} \, E_{ir} \, n_i(r_i) \, n_r(r_r) \, a_r(r_r) \, m_r(r_r)  \, v_{term}(r_r) \, d r_i d r_r
\label{eq:accr_ir}
\end{equation}
```
where:
 - ``E_{ir}`` is the collision efficiency between rain and cloud ice
 - ``n_i`` and ``n_r`` are the cloud ice and rain size distributions
 - ``m_r``, ``a_r`` and ``v_{term}`` are the mass(radius), cross section(radius) and terminal velocity(radius) relations for rain
 - ``r_i`` and ``r_r`` mark integration over cloud ice and rain size distributions

Integrating eq.(\ref{eq:accr_ir}) yields:
```math
\begin{equation}
\left. \frac{d \, q_{rai}}{dt} \right|_{accr\_ri} =
  - E_{ir} \, n_0^{rai} \, n_0^{ice} \, \alpha^{rai} \, \gamma^{rai} \, \zeta^{rai} \frac{1}{\lambda_{ice}}
  \frac{\Gamma(\beta^{rai} + \delta^{rai} + \eta^{rai} +1)}{\lambda_{rai}^{(\beta^{rai} + \delta^{rai} + \eta^{rai} + 1)}}
\end{equation}
```

Collisions between rain and snow result in snow in temperatures below freezing andin rain in temperatures above freezing.
The source term is defined as:
```math
\begin{equation}
\left. \frac{d \, q_i}{dt} \right|_{accr} =
    \int_0^\infty \int_0^\infty
    n_i(r_i) \, n_j(r_j) \, a(r_i, r_j) \, m_j(r_j) \, E_{ij} \, \left|v_{term}(r_i) - v_{term}(r_j)\right| \, dr_i dr_j
\label{eq:accr_sr1}
\end{equation}
```
where
- ``i`` stands for rain (``T>T_{freezing}``) or snow (``T<T_{freezing}``)
- ``j`` stands for snow (``T>T_{freezing}``) or rain (``T<T_{freezing}``)
- ``a(r_i, r_j)`` is the crossection of the two colliding particles

There are two additional assuptions that we make to integrate eq.(\ref{eq:accr_sr1}):
- ``\left|v_{term}(r_i) - v_{term}(r_j)\right| \approx \left| v_{ti} - v_{tj} \right|``
  We approximate the teminal velocity difference for each particle pair with
  a difference between mass-weighted mean terminal velocityes and move it outside of the integral.
  See the discussion in [Ikawa\_and\_Saito\_1991](https://www.mri-jma.go.jp/Publish/Technical/DATA/VOL_28/28_005.pdf) page 88.

-  We assume that ``a(r_i, r_j) = \pi (r_i + r_j)^2``.
   This corresponds to a geometric formulation of the collision kernel, aka cylindrical fomulation,
   see [Wang\_et\_al\_2005](https://journals.ametsoc.org/doi/pdf/10.1175/JAS3655.1) for discussion.


The eq.(\ref{eq:accr_sr1}) can then be integrated as:
```math
\begin{align}
\left. \frac{d \, q_i}{dt} \right|_{accr} & =
    \pi \, n_0^{i} \, n_0^{j} \, \alpha_j \, E_{ij} \left| v_{ti} - v_{tj} \right|
    \int_0^\infty \int_0^\infty
    (r_i + r_j)^2
    r_{j}^{\beta_j} \,
    exp(- \lambda_j r_j) \,
    exp(- \lambda_i r_i) \,
    dr_i dr_j \\
    & =
    \pi \, n_0^{i} \, n_0^{j} \, \alpha_j \, E_{ij} \left| v_{ti} - v_{tj} \right|
    \left(
        \frac{2 \Gamma(\beta_j + 1)}{\lambda_i^3 \lambda_j^{\beta_j+1}}
        + \frac{2 \Gamma(\beta_j + 2)}{ \lambda_i^2 \lambda_j^{\beta_j+2}}
        + \frac{\Gamma(\beta_j + 3)}{\lambda_i \lambda_j^{\beta_j + 3}}
    \right)
\end{align}
```

!!! note
    Both of the assumptions needed to integrate the snow-rain accretion rate could be revisited:

    The discussion on page 88 in
    [Ikawa\_and\_Saito\_1991](https://www.mri-jma.go.jp/Publish/Technical/DATA/VOL_28/28_005.pdf)
    suggests an alternative approximation of the velocity difference.

    The ``(r_i + r_j)^2`` assumption for the crossection is inconsistent with the
    snow crossection used when modelling collisions with cloud water and cloud ice.


### Rain evaporation and snow sublimation

We start from a similar equation as when computing snow autoconversion rate
but integrate it from ``0`` to ``\infty``.

```math
\begin{equation}
  \left. \frac{dq}{dt} \right|_{evap\_subl} = \frac{1}{\rho} \int_{0}^{\infty} \frac{dm(r)}{dt} n(r) dr
\end{equation}
```
In contrast to eq.(\ref{eq:mass_rate}), now we are taking into account ventilation effects:

```math
\begin{equation}
  \frac{dm}{dt} = 4 \pi \, r \, (S - 1) \, G(T) \, F(r)
\label{eq:mass_rate2}
\end{equation}
```
where:
 - ``F(r)`` is the rain drop ventilation factor defined in (\ref{eq:ventil_factor})
 - supersaturation S is computed over water or ice
 - ``L`` is the latent heat of vaporization or sublimation.

The final integral is:

```math
\begin{align}
\left. \frac{dq}{dt} \right|_{evap\_subl} & =
    \frac{4 \pi n_0}{\rho} (S - 1) G(T)
    \int_0^\infty
    \left(
       a_{vent} \, r +
       b_{vent} \left( \frac{\nu_{air}}{D_{vapor}} \right)^{\frac{1}{3}}
         \left( \frac{2 \zeta}{\nu_{air}} \right)^{\frac{1}{2}} r^{\frac{\eta + 3}{2}}
    \right)
    exp(-\lambda r) dr \\
    & =
    \frac{4 \pi n_0}{\rho} (S - 1) G(T) \lambda^{-2}
    \left(
       a_{vent} +
       b_{vent} \left( \frac{\nu_{air}}{D_{vapor}} \right)^{\frac{1}{3}}
         \left( \frac{2 \zeta}{\nu_{air}} \right)^{\frac{1}{2}}
         \Gamma \left( \frac{\eta + 5}{2} \right) \lambda^{- \frac{\eta +1}{2}}
    \right)
\end{align}
```

!!! note
    We should take into account the non-spherical snow shape. - Modify the Reynolds number and growth equation.

### Snow melt

If snow occurs in temperatures above freezing it is melting into rain.
The sink for snow is parameterized again as
```math
\begin{equation}
  \left. \frac{dq}{dt} \right|_{melt} = \frac{1}{\rho} \int_{0}^{\infty} \frac{dm(r)}{dt} n(r) dr
\end{equation}
```
For snow melt
```math
\begin{equation}
  \frac{dm}{dt} = 4 \pi \, r \, \frac{K_{thermo}}{L_f} (T - T_{freeze}) \, F(r)
\label{eq:mass_rate3}
\end{equation}
```
where:
 - ``F(r)`` is the ventilation factor defined in (\ref{eq:ventil_factor})
 - ``L_f`` is the latent heat of freezing.

```math
\begin{equation}
\left. \frac{dq}{dt} \right|_{evap\_subl} =
    \frac{4 \pi \, n_0 \, K_{thermo}}{\rho \, L_f} (T - T_{freeze}) \lambda^{-2}
    \left(
       a_{vent} +
       b_{vent} \left( \frac{\nu_{air}}{D_{vapor}} \right)^{\frac{1}{3}}
         \left( \frac{2 \zeta}{\nu_{air}} \right)^{\frac{1}{2}}
         \Gamma \left( \frac{\eta + 5}{2} \right) \lambda^{- \frac{\eta +1}{2}}
    \right)
\end{equation}
```
