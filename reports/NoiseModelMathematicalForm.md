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

  - As the name indicates, this is a combination of a phase flip and a bit flip, since $Y=iXZ$.

### Depolarization channel

- The depolarization channel is an important type of quantum noise. Imagine we take a single qubit, and with probability $p$ that qubit is depolarized. That is it is replaced by the completely mixed state, $I/2$. With probability $1-p$ the qubit is left untouched. The state of the quantum system after this noise is

- $$
  \varepsilon(\rho) = \frac{pI}{2}+(1-p)\rho
  $$

- A quantum circuit simulating the depolarizing channel is illustrated as follows:

- ![image-20241111162652716](/home/arthur/.config/Typora/typora-user-images/image-20241111162652716.png)

- The top line of the circuit is the input to the depolarizing channel, while the bottom two lines are an 'environment'  to simulate the channel. We have used an environment with two mixed state inputs. The idea is that the third qubit, initially a mixture of the state $\ket{0}$ with probability $1-p$ and state $\ket{1}$ with probability $p$ acts as a control for whether or not the completely mixed state $I/2$ stored in the second qubit is swapped into the first qubit.

- The formula above is not in the operator-sum representation. However, if we observe that for arbitrary $\rho$,

- $$
  \frac{I}{2} = \frac{\rho+X\rho X+Y\rho Y+Z\rho Z}{4}
  $$

- and then substitute for $I/2$ into formula (1) we arrive at the equation

- $$
  \varepsilon(\rho)=(1-\frac{3p}{4})\rho+\frac{p}{4}(X\rho X+Y\rho Y + Z\rho Z)
  $$

- showing that  

