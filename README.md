# wake-e2e — trained model and a reference case

Released with

> Ye, M., Wei, D., Wang, N., Wan, D. *A physics-aware end-to-end data-driven framework for
> efficient and accurate wind-turbine wake prediction in large wind farms.*

This repository contains the trained model of the paper and one FAST.Farm case, so that the
prediction and the reference field can be inspected directly. Nothing else is redistributed.

```
models/p1_rss.pt   trained parameters, 4 609 in total (PyTorch state dict)
data/P1_L08_C1.npz one FAST.Farm case: time-mean hub-height field
```

## The model

A multilayer perceptron with two hidden layers of 64 units, evaluated at coordinates measured from
the centre of each turbine and shared by all of them. Its input is the pair
$(\Delta x/D, \Delta y/D)$ together with three scalars that describe the inflow (wind speed
normalised by its rated value, turbulence intensity, shear exponent); its output is the deficit of
that turbine. The deficits are composed by the root-sum-square rule of the flow solver, which
contains no trainable coefficient. The network therefore predicts the field of any layout with the
same weights, and the parameter count does not depend on the number of turbines.

## The case

`P1_L08_C1.npz` holds three arrays:

| key | shape | meaning |
|---|---|---|
| `U_mean` | 141 × 179 | time-mean axial velocity, m/s, on the hub-height plane |
| `x` | 179 | streamwise grid, m |
| `y` | 141 | lateral grid, m |

The case is layout L08 with inflow condition C1: nine NREL 5-MW turbines, 11.4 m/s, 6 % turbulence
intensity and a shear exponent of 0.14. The layout coordinates are given in the paper (Fig. 3) and
in the case matrix (Table 3). The field was produced with FAST.Farm v5.0.0.

## Using the parameters

The state dict expects `torch.nn.Linear` layers of shape 5→64→64→1 with `tanh` activations between
them, and the inputs are normalised as described in Section 3.4 of the paper. The evaluation scripts,
and the remaining cases, are available from the corresponding author on request.

## Licence

Parameters and data: CC BY 4.0.
