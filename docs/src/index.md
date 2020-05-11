# ClimateMachine

```@meta
CurrentModule = ClimateMachine
```

`ClimateMachine` is a software package that models the evolution of the Earth system over weeks to centuries.
`ClimateMachine` solves three-dimensional partial differential equations for the distributions of water, momentum, energy,
and tracers such as carbon in the atmosphere, oceans, and on land.

`ClimateMachine` will harness a wide range of Earth observations and data generated computationally to predict the evolution of Earth’s climate and features such as
droughts, rainfall extremes, and high-impact storms.

## Subcomponents

`ClimateMachine` currently consists of three models for the subcomponents of the Earth system:
 
* `ClimateMachine.Atmos`: A model of the fluid mechanics of the atmosphere and its interaction with solar radiation and phase changes of water that occur, for example, in clouds.
* `ClimateMachine.Ocean`: A model for the fluid mechanics of the ocean and its distributions of heat, salinity, carbon, and other tracers.
* `ClimateMachine.Land`: A model for the flow of energy and water in soils and on the land surface, for the biophysics of vegetation on land, and for the transfer and storage of carbon in the land biosphere.

The subcomponents will be coupled by exchanging water, momentum, energy, and tracers such as carbon dioxide across their boundaries.

## Dynamical core

A dynamical core based on discontinuous Galerkin numerical methods is used to discretize the physical conservation laws that underlie each of `ClimateMachine`'s subcomponents.

## Authors

`ClimateMachine` is being developed by [the Climate Modeling Alliance](https://clima.caltech.edu).

