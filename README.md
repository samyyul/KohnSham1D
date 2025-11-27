# KohnSham1D
Short Project writing Kohn-Sham Density Functional Theory (KS-DFT) code.

Source Material: http://dcwww.camd.dtu.dk/~askhl/files/python-dft-exercises.pdf

In this project, we produce a 1D DFT code for the energy of a harmonic oscillator, breaking down the Hamiltonain to describe the total energy of the system. 

The Hamiltonian is sufficiently represented through calculations of electron density, KS wavefunctions and a series of potential.


$$
\hat{H} = -\frac{1}{2}\frac{d^{2}}{dx^{2}} + v_{\mathrm{Ha}}(x) + v_{X}^{\mathrm{LDA}}(x) + x^{2}
$$

We can segment this hamiltonian into multiple functions.

## Harmonic Oscillator

The harmonic oscillator for a non-interacting particle is given as:

$$
\hat{H} = \hat{T} =
    -\frac{1}{2}\frac{d^{2}}{dx^{2}} + x^{2}
$$

Where $$\hat{T}$$ is the kinetic energy operator.

## Coulomb Potential.

$$
E_{\mathrm{Ha}}[n]
  = \frac{1}{2}\int\\\int
    \frac{n(x)\,n(x')}{|x-x'| + 1}\,dx\,dx'.
$$

$$
v_{\mathrm{Ha}}(x)
  = \int \frac{n(x')}{|x-x'| + 1}\,dx'.
$$

## Exchange Potential.

$$
E_{X}[n]
  = -\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3}
    \int n(x)^{4/3}\,dx .
$$

$$
v_{X}^{\mathrm{LDA}}(x)
  = -\left(\frac{3}{\pi}\right)^{1/3}
    n(x)^{1/3}.
$$

## Total Energy


The total KS energy can be seperated into kinetic energy, Coulomb and exchange Potential energies, and correlation energy. 

$$
E[n] =
\sum_{n} f_n\,\epsilon_n
\-\
\int n(x)\,v(x)\,dx
\+\
E_{\mathrm{Ha}}[n]
\+\
E_{X}[n]
\+\
v_{\mathrm{exc}}(x)
$$

For simplicity, we ignore correlation, such that:

$$
v_{\mathrm{exc}}(x) = v_{X}^{\mathrm{LDA}}(x).
$$
