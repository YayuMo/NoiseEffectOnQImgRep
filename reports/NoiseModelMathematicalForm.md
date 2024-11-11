# Report 3

## Problem Definition



## Noise Type

### Bit Flip and Phase Flip channels

- **Bit Flip**

  - The bit flip channel flips the state of a qubit from $\ket{0}$ to $\ket{1}$ (and vice versa) with probability $1-p$. It has operation elements:

  - $$
    E_0 = \sqrt{p}I=\sqrt{p}\begin{bmatrix}
    1 & 0 \\
    0 & 1
    \end{bmatrix} \quad 
    E_1 = \sqrt{1-p}X=\sqrt{1-p}\begin{bmatrix}
    0 & 1 \\
    1 & 0
    \end{bmatrix}
    $$

- **Phase Flip**

  - The phase flip channel has operation elements

  - $$
    E_0 = \sqrt{p}I=\sqrt{p}\begin{bmatrix}
    1 & 0 \\
    0 & 1
    \end{bmatrix} \quad 
    E_1 = \sqrt{1-p}Z=\sqrt{1-p}\begin{bmatrix}
    1 & 0 \\
    0 & -1
    \end{bmatrix}
    $$

  - As a special case of the phase flip channel, consider the quantum operation which arises when we choose $p=1/2$. Using the freedom in the operator-sum representation this operation may be written 

  - $$
    \rho = \varepsilon(\rho) = P_0\rho P_0 + P_1\rho P_1
    $$

  - where $P_0=\ket{0}\bra{0},P_1=\ket{1}\bra{1}$, which corresponds to a measurement of the qubit in the $\ket{0}, \ket{1}$ basis, with the result of the measurement unknown. Using the above prescription it is easy to see that the corresponding map on the Bloch sphere is given by 

  - $$
    (r_x, r_y, r_z) \rarr (0, 0, r_z)
    $$

- **Bit-Phase Flip**

  - The bit-phase flip channel has operation elements

  - $$
    E_0 = \sqrt{p}I=\sqrt{p}\begin{bmatrix}
    1 & 0 \\
    0 & 1
    \end{bmatrix} \quad 
    E_1 = \sqrt{1-p}Y=\sqrt{1-p}\begin{bmatrix}
    0 & -i \\
    i & 0
    \end{bmatrix}
    $$

  - 