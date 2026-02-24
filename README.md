# Python + WebGPU Port of Gpufit

This repository contains an experimental Python + WebGPU port of the original **Gpufit** project — a GPU‑accelerated Levenberg–Marquardt curve‑fitting library originally implemented in C++ and CUDA.  
The port explores how modern GPU compute APIs (WebGPU) and Python can reproduce or adapt parts of Gpufit’s functionality.  
Most of the code was developed through *vibecoding* (AI‑assisted iterative coding).

This project is part of the **2026 BIOMICS Hackathon**:  
👉 https://biomics.bacpop.org

---

## ⭐ About the Original Gpufit

The official Gpufit repository is available here:  
👉 https://github.com/gpufit/ [1](https://innoter.com/en/articles/hyperspectral-imaging/)

Gpufit is an open‑source GPU‑accelerated toolkit for fast curve fitting using the Levenberg–Marquardt algorithm. It was developed at the Max Planck Institute for Biophysical Chemistry and provides C, Python, Java, and MATLAB interfaces.  
According to the authors, Gpufit achieves up to **42× speed‑up** over equivalent CPU implementations with no loss of precision. [2](https://www.prophotonix.com/blog/hyperspectral-vs-multispectral-imaging-whats-the-difference/)

### 📄 Original Research Paper

If you use this work or the concepts behind it, please cite the original Gpufit publication:

> **Adrian Przybylski, Björn Thiel, Jan Keller‑Findeisen, Bernd Stock & Mark Bates**  
> *Gpufit: An open-source toolkit for GPU-accelerated curve fitting.*  
> Scientific Reports, 2017.  
> DOI: 10.1038/s41598-017-15405-8  
> (Open‑access version available from Scientific Reports)  
> [1](https://innoter.com/en/articles/hyperspectral-imaging/)[2](https://www.prophotonix.com/blog/hyperspectral-vs-multispectral-imaging-whats-the-difference/)

This port is an independent experimental adaptation and is **not** a drop‑in replacement for the official CUDA‑based implementation.

---

## 🚧 About This Port

This repository includes:

- A **Python re‑implementation** of selected Gpufit components  
- A **WebGPU backend** using compute shaders  
- A heavily *vibecoded* codebase, meaning:
  - Exploratory prototyping  
  - AI‑assisted coding  
  - Rapid iteration over design choices  

### ⚠️ Limitations

- Not all original Gpufit features are implemented  
- WebGPU kernels are experimental and not fully optimized  
- The API may diverge from the original  
- Expect breaking changes and evolving structure
- **Even this README was vibe coded**, I'm just experimenting with it hehe. 

This port is intended for research, experimentation, and as a demonstration of how curve fitting could be adapted to WebGPU.
