# FlowChart

Stepping a balance law in time involves calling a "`compute_right_hand_side!`" or "`compute_tendencies!`" function. As described[^1], this function calls several other functions including

 - `update_aux!`
 - `flux_diffusive!`
 - `flux_nondiffusive!`
 - `source!`

Below is a flow chart to provide a more detailed picture of the order these functions are called:


```@example tendencies_flow_chart
using TikzGraphs
using LightGraphs
g = DiGraph(4)
TikzGraphs.plot(g)

savefig("tendencies_flow_chart.svg") # hide
nothing # hide
```
![](tendencies_flow_chart.svg)


[^1] [How to build a balance law]()