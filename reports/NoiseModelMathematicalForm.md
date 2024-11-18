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

- showing that depolarizing channel has operation elements $\{ \sqrt{1-3p/4}I,\sqrt{p}X/2, \sqrt{p}Y/2, \sqrt{p}Z/2 \}$. Note, incidentally, that it is frequently convenient to parameterize the depolarizing channel in different ways, such as

- $$
  \varepsilon(\rho)=(1-p)\rho + \frac{p}{3}(X\rho X+Y \rho Y+Z\rho Z)
  $$

- which has the interpretation that the state $\rho$ is left alone with probability $1-\rho$, and the operators $X, Y$ and $Z$ applied each with probability $p/3$. 

- The depolarizing channel can, of course, be generalized to quantum system of dimension more than two. For a $d$-dimensional quantum system the depolarizing channel again replaces the quantum system with the completely mixed state $I/d$ with probability $p$, and leaves the state untouched otherwise. The corresponding quantum operation is 

- $$
  \varepsilon(\rho) = \frac{pI}{d}+(1-p)\rho
  $$

### Amplitude Damping channel

- An important application of quantum operations is the description of $energy \ dissipation$ - effects due to loss energy from a quantum system. What are the dynamics of an atom which is spontaneously emitting a photon? How does a spin system at high temperature approach equilibrium with its environment? What is the state of a photon in an interferometer or cavity when it is subject to scattering and attenuation?

- Each of these processes has its own unique features, but the general behavior of all of them is well characterized by a quantum operation known as $amplitude \ damping$, which we can derive by considering the following scenario. Suppose we have a single optical mode containing the quantum state $a\ket{0}+b\ket{1}$, a superposition of zero or one photons. The scattering of a photon from this mode can be modeled by thinking of inserting a partially slivered mirror, a beamsplitter, in the path of the photon. As we saw in Section 7.4.2, this beamsplitter allows the photon to couple to couple to another single optical mode (representing the environment), according to the unitary transformation $B=exp[\theta(a^{\dagger}b-ab^{\dagger})]$, where $a, a^{\dagger}$ and $b, b^{\dagger}$ are annihilation and creation operators for photons in the two modes. The output after the beamsplitter, assuming the environment starts out with no photons, is simply $B\ket{0}(a\ket{0}+b\ket{1})=a\ket{00}+b(\cos \theta\ket{01}+\sin \theta \ket{10})$, using Equation (7.34). Tracing over the environment gives us the quantum operation

- $$
  \varepsilon_{AD}(\rho)=E_0\rho E_0^{\dagger}+E_1\rho E_1^{\dagger}
  $$

- where $E_k=\bra{k}B\ket{0}$ are

- $$
  E_0=\begin{bmatrix}
  1 & 0 \\
  0 & \sqrt{1-\gamma}
  \end{bmatrix} \\
  E_1=\begin{bmatrix}
  0 & \sqrt{\gamma} \\
  0 & 0
  \end{bmatrix}
  $$

- the operation elements for amplitude damping. $\gamma=\sin^2\theta$ can be thought of as the probability of losing a photon.

- Observe that no linear combination can be made of $E_0$ and $E_1$ to give an operation element proportional to the identity (though compare with Exercise 8.23). The $E_1$ operation changes a $\ket{1}$ state into a $\ket{0}$ state, corresponding to the physical process of losing a quantum of energy to the environment. $E_0$ leaves $\ket{0}$ unchanged, but reduces the amplitude of a $\ket{1}$ state; physically, this happens because a quantum of energy was not lost to the environment, and thus the environment now perceives it to be more likely that the system is in the $\ket{0}$ state, rather than the $\ket{1}$ state.

- A general characteristic of a quantum operation is the set of states that are left invariant under the operation. For example, we have seen how the phase flip channel leaves the $\hat{z}$ axis of the Bloch sphere unchanged; this corresponds to states of the form $p\ket{0}\bra{0}+(1-p)\ket{1}\bra{1}$ for arbitrary probability $p$. In the case of amplitude damping, only the ground state $\ket{0}$ is left invariant. That is a natural consequence of our modeling the environment as starting in the $\ket{0}$ state, as if it were at zero temperature.

- What quantum operation describes the effect of dissipation to an environment at finite temperature? This process, $\varepsilon_{GAD}$, called $generalized \ amplitude \ damping$, is defined for single qubits by the operation elements

- $$
  E_0 = \sqrt{p}\begin{bmatrix}
  1 & 0 \\
  0 & \sqrt{1-\gamma}
  \end{bmatrix}\\
  E_1 = \sqrt{p}\begin{bmatrix}
  0 & \sqrt{\gamma} \\
  0 & 0
  \end{bmatrix}\\
  E_2 = \sqrt{1-p}\begin{bmatrix}
  \sqrt{1-\gamma} & 0 \\
  0 & 1
  \end{bmatrix}\\
  E_3 = \sqrt{1-p}\begin{bmatrix}
  0 & 0 \\
  \sqrt{\gamma} & 0
  \end{bmatrix}
  $$

- where the stationary state

- $$
  \rho_{\infin}=\begin{bmatrix}
  p & 0 \\
  0 & 1-p
  \end{bmatrix}
  $$

- satisfies $\varepsilon_{GAD}(\rho_{\infin})$. Generalized amplitude damping describes the '$T_1$' relaxation process due to coupling of spins to their surrounding lattice, a large system which is in thermal equilibrium at a temperature often much higher than the spin temperature. This is the case relevant to NMR quantum computation, where some of the properties of $\varepsilon_{GAD}$ described in Box 8.3 become important.

- We can visualize the effect of amplitude damping in the Bloch representation as the Bloch vector transformation

- $$
  (r_x, r_y, r_z) \rarr (r_x\sqrt{1-\gamma}, r_y\sqrt{1-\gamma}, \gamma+r_z(1-\gamma))
  $$

- When $\gamma$ is replaced with a time-varying function like $1-e^{-t/T_1}$ ($t$ is time, and $T_1$ just some constant characterizing the speed of the process), as is often the case for real physical processes, we can visualize the effects of amplitude damping as a $flow$ on the Bloch sphere, which moves every point in the unit ball towards a fixed point at the north pole, where $\ket{0}$ is located. This is shown in Figure 8.14.

- Similarly, generalized amplitude damping performs the transformation

- $$
  (r_x, r_y, r_z) \rarr (r_x\sqrt{1-\gamma}, r_y\sqrt{1-\gamma}, \gamma(2p-1)+r_z(1-\gamma))
  $$

- Comparing last eq and this eq, it is clear that amplitude damping and generalized amplitude damping differ only in the location of the fixed point of the flow; the final state is along the $\hat{z}$ axis, at the point $(2p-1)$, which is a mixed state.

- **Box 8.3: Generalized amplitude damping and effective pure states**

  - The notion of 'effective pure states' introduced in Section 7.7 was found to be useful in NMR implementations of quantum computers. These states behave like pure states under unitary evolution and measurement of traceless observables. How do they behave under quantum operations? In general, non-unitary quantum operations ruin the effectiveness of these states, but surprisingly, they can behave properly under generalized amplitude damping.

  - Consider a single qubit effective pure state $\rho=(1-p)I+(2p-1)\ket{0}\bra{0}$. Clearly, traceless measurement observables acting on $U\rho U^{\dagger}$ produce results which are proportional to those on the pure state $U\ket{0}\bra{0}U^{\dagger}$. Suppose $\rho$ is the stationary state of $\varepsilon_{GAD}$. Interestingly, in this case,

  - $$
    \varepsilon_{GAD}(U\rho U^{\dagger})=(1-p)I+(2p-1)\varepsilon_{AD}(U\rho U^{\dagger})
    $$

  - That is, under generalized amplitude damping, an effective pure state can remain such, and moreover, the 'pure' component of $\rho$ behaves as if it were undergoing amplitude damping to a reservoir at zero temperature!

### Phase Damping channel

- A noise process that is uniquely quantum mechanical, which describes the loss of quantum information without loss of energy, is $phase \ damping$. Physically it describes, for example, what happens when a photon scatters randomly as it travels through a waveguide, or how electronic states in an atom are perturbed upon interacting with distant electrical charges. The energy eigenstates of a quantum system do not change as a function of time, but do accumulate a phase which is not precisely known, partial information about this quantum phase - the $relative$ phases between the energy eigenstates - is lost.

- A very simple model for this kind of quantum noise is the following. Suppose that we have a qubit $\ket{\psi}=a\ket{0}+b\ket{1}$ upon which the rotation operation $R_z(\theta)$ is applied, where the angle of rotation $\theta$ is random. The randomness could originate, for example, from a deterministic interaction with an environment, which never again interacts with the system and thus is implicitly measured (see Section 4.4). We shall call this random $R_z$ operation a $phase \ kick$. Let us assume that the phase kick angle $\theta$ is well represented as a random variable which has a Gaussian distribution with mean 0 and variance $2\lambda$.

- The output state from this process is given by the density matrix obtained from averaging over $\theta$,

- $$
  \rho = \frac{1}{\sqrt{4\pi\lambda}}\int^{\infin}_{-\infin}R_z(\theta)\ket{\psi}\bra{\psi}R_z^{\dagger}(\theta)e^{-\theta^2/4\lambda}d\theta \\
  =\begin{bmatrix}
  |a|^2 & ab^*e^{-\lambda} \\
  a^*be^{-\lambda} & |b|^2
  \end{bmatrix}
  $$

- The random phase kicking causes the expected value of the off-diagonal elements of the density matrix to decay exponentially to zero with time. That is a characteristic result of phase damping.

- Another way to derive the phase damping quantum operation is to consider an interaction between two harmonic oscillators, in a manner similar to how amplitude damping was derived in the last section, but this time with the interaction Hamiltonian

- $$
  H = \chi a^{\dagger}a(b+b^{\dagger})
  $$

- Letting $U=exp(-iH\Delta t)$, considering only the $\ket{0}$ and $\ket{1}$ states of the $a$ oscillator as our system, and taking the environment oscillator to initially be $\ket{0}$, we find that tracing over the environment gives the operation elements $E_k=\bra{k_b}U\ket{0_b}$, which are

- $$
  E_0 = \begin{bmatrix}
  1 & 0 \\
  0 & \sqrt{1-\lambda}
  \end{bmatrix} \\
  E_1 = \begin{bmatrix}
  0 & 0 \\
  0 & \sqrt{\lambda}
  \end{bmatrix}
  $$

- where $\lambda=1-\cos^2(\chi\Delta t)$ can be interpreted as the probability that a photon from the system has been scattered (without loss of energy). As was the case for amplitude damping, $E_0$ leaves $\ket{0}$ unchanged, but reduces the amplitude of a $\ket{1}$ state; unlike amplitude damping, however, the $E_1$ operation destroys $\ket{0}$ and reduces the amplitude of the $\ket{1}$ state, and does not change it into a $\ket{0}$.

- By applying Theorem 8.2, the unitary freedom of quantum operations, we find that a unitary recombination of $E_0$ and $E_1$ gives a new set of operation elements for phase damping,

- $$
  \tilde{E_0}=\sqrt{\alpha}\begin{bmatrix}
  1 & 0 \\
  0 & 1
  \end{bmatrix} \\
  \tilde{E_1} = \sqrt{1-\alpha}\begin{bmatrix}
  1 & 0 \\
  0 & -1
  \end{bmatrix}
  $$

- where $\alpha=(1+\sqrt{1-\lambda})/2$. Thus the phase damping quantum operation is $exactly$ the same as the phase flip channel which we encountered in Section 8.3.3!

- Since phase damping is the same as the phase flip channel, we have already seen how it is visualized on the Bloch sphere, in Figure 8.9. This corresponds to the Bloch vector transformation

- $$
  (r_x, r_y, r_z) \rarr (r_x\sqrt{1-\lambda}, r_y\sqrt{1-\lambda}, r_z)
  $$

- which has the effect of shrinking the sphere into ellipsoids. Phase damping is often referred to as a '$T_2$' (or 'spin-spin') relaxation process, for historical reasons, where $e^{-t/2T_2}=\sqrt{1-\lambda}$. As a function of time, the amount of damping increases, corresponding to an inwards flow of all points in the unit ball towards the $\hat{z}$ axis. Note that states along the $\hat{z}$ axis remain invariant.

- Historically, phase damping was a process that was almost always thought of, physically, as resulting from a random phase kick or scattering process. It was not until the connection to the phase flip channel was discovered that quantum error-correction was developed, since it was thought that phase errors were $continuous$ and couldn't be described as a discrete process! In fact, single qubit phase errors can $always$ be thought of as resulting from a process in which either nothing happens to a qubit, with probability $\alpha$, or with probability $1-\alpha$, the qubit is flipped by the $Z$ Pauli operation. Although this might not be the actual microscopic physical process happening, from the standpoint of the transformation occurring to a qubit over a discrete time interval large compared to the underlying random process, there is no difference at all.

- Phasing damping is one of the most subtle and important processes in the study of quantum computation and quantum information. It has been the subject of an immense amount of study and speculation, particularly with regard to why the world around us appears to be so classical, with superposition states not a part of our everyday experience! Perhaps it is phase damping that is responsible for this absence of superposition states from the everyday (Excercise 8.31)? The pioneering quantum physicist Schrodinger was perhaps the first to pose this problem and he did this in a particularly stark form, as discussed in Box 8.4. 

### Summary of Quantum noise and quantum operations

- **The operator-sum representation:** The behavior of an open quantum system can be modeled as 

  - $$
    \varepsilon(\rho)=\sum_kE_k\rho E_k^{\dagger}
    $$

  - where $E_k$ are operation elements, satisfying $\sum_kE_k^{\dagger}E_k=I$ if the quantum operation is trace-preserving.

- **Environmental models for quantum operations:** A trace-preserving quantum operation can always be regarded as arising from the unitary interaction of a system with an initially uncorrelated environment, and vice versa. Non-trace-preserving quantum operations may be treated similarly, except an additional projective measurement is performed on the composite of system and environment, with the different outcomes corresponding to different non-trace-preserving quantum operations.

- **Quantum process tomography:** A quantum operation on a $d$-dimensional quantum system can be completely determined by experimentally measuring the output density matrices produced from $d^2$ pure state inputs.

- **Operation elements for important single qubit quantum operations:**

  - depolarizing channel

    - $$
      \sqrt{1-\frac{3p}{4}}\begin{bmatrix}
      1 & 0 \\
      0 & 1
      \end{bmatrix}, \quad
      \sqrt{\frac{p}{4}}\begin{bmatrix}
      0 & 1 \\
      1 & 0
      \end{bmatrix}, \quad \\
      \sqrt{\frac{p}{4}}\begin{bmatrix}
      0 & -i \\
      i & 0
      \end{bmatrix}, \quad
      \sqrt{\frac{p}{4}}\begin{bmatrix}
      1 & 0 \\
      0 & -1
      \end{bmatrix},
      $$

  - amplitude damping

    - $$
      \begin{bmatrix}
      1 & 0 \\
      0 & \sqrt{1-\gamma}
      \end{bmatrix}, \quad
      \begin{bmatrix}
      0 & \sqrt{\gamma} \\
      0 & 0
      \end{bmatrix}
      $$

  - phase damping

    - $$
      \begin{bmatrix}
      1 & 0 \\
      0 & \sqrt{1-\gamma}
      \end{bmatrix}, \quad
      \begin{bmatrix}
      0 & 0 \\
      0 & \sqrt{\gamma}
      \end{bmatrix}
      $$

  - phase flip

    - $$
      \sqrt{p}\begin{bmatrix}
      1 & 0 \\
      0 & 1
      \end{bmatrix}, \quad
      \sqrt{1-p}\begin{bmatrix}
      1 & 0 \\
      0 & -1
      \end{bmatrix}
      $$

  - bit flip

    - $$
      \sqrt{p}\begin{bmatrix}
      1 & 0 \\
      0 & 1
      \end{bmatrix}, \quad
      \sqrt{1-p}\begin{bmatrix}
      0 & 1 \\
      1 & 0
      \end{bmatrix}
      $$

  - bit-phase flip

    - $$
      \sqrt{p}\begin{bmatrix}
      1 & 0 \\
      0 & 1
      \end{bmatrix}, \quad
      \sqrt{1-p}\begin{bmatrix}
      0 & -i \\
      i & 0
      \end{bmatrix}
      $$





