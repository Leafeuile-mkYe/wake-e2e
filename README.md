# wake-e2e — trained model and a reference case

This repository is created by Dr. Maokun Ye @ Shanghai, China

Released with

> Ye, M, et.al. *A physics-aware end-to-end data-driven framework for
> efficient and accurate wind-turbine wake prediction in large wind farms.*

This repository contains the trained model of the paper and one FAST.Farm case, so that the
prediction and the reference field can be inspected directly. Nothing else is redistributed.

```
models/p1_rss.pt   trained parameters, 4 609 in total
data/P1_L08_C1.npz one FAST.Farm case: time-mean hub-height field
predict.py         loads the parameters and evaluates them on that case
```

Run it with

```bash
python -m pip install numpy torch
python predict.py                 # prints the error of Table 6 for this case
python predict.py --png cmp.png   # also writes a model/reference comparison
```

On the released case the script reproduces the published value, 0.247 m/s or 2.2 % of the inflow
velocity.

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

The file is a Python dictionary:

| key | content |
|---|---|
| `state_dict` | `mlp.0` 5→64, `mlp.2` 64→64, `mlp.4` 64→1, with SiLU between the layers |
| `hidden` | 64 |
| `n_cond` | 3 |
| `u_ref` | 11.4 (m/s, the speed used to normalise the inflow) |
| `steps` | 40 000 |

The output of the last layer is passed through a softplus and clipped at 0.75, which gives the
deficit of one turbine; the deficits of the nine turbines are then composed by the fixed
root-sum-square rule and subtracted from the inflow. The inputs are normalised as described in
Section 3.4 of the paper. The remaining cases are available from the corresponding author on request.

## Licence

Parameters and data: CC BY 4.0.
