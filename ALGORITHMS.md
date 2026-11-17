# VideoUpscale Classical Computer-Vision Algorithms

All algorithms in VideoUpscale are purely mathematical and classical signal-processing methods (zero neural networks, zero pretrained models).

---

## 1. Frame & Degradation Analysis (`src/analysis/frame_analysis.cpp`)

All analysis operates on normalized 8-bit luma $Y \in [0, 1]$.

### 1.1 Noise Variance Estimation (Immerkær MAD)
Convolves $Y$ with the $3 \times 3$ zero-mean Laplacian noise operator:
$$N = \begin{bmatrix} 1 & -2 & 1 \\ -2 & 4 & -2 \\ 1 & -2 & 1 \end{bmatrix}$$
For additive white Gaussian noise $\mathcal{N}(0, \sigma^2)$, $(Y * N) \sim \mathcal{N}(0, 36\sigma^2)$. To avoid inflating $\hat{\sigma}$ on sharp edges or textures, we restrict sampling to pixels below the 55th percentile of Sobel gradient magnitude and compute the robust Median Absolute Deviation:
$$\hat{\sigma} = \frac{1.4826}{6}\,\mathrm{median}(|Y * N|)$$

### 1.2 Blur Estimation (Crete Re-Blurring)
Compares horizontal and vertical absolute pixel differences of $Y$ against 9-tap box-blurred versions $B_h, B_v$. If $Y$ is already blurred, additional blurring changes local variations very little; the normalized difference yields a blur score in $[0, 1]$ (higher = blurrier).

### 1.3 Blockiness Estimation (8×8 Boundary Excess)
Measures mean absolute differences across 8-pixel boundaries ($B$) versus interior pixel steps ($I$), returning $\max(0, (B - I)/(B + I + \epsilon)) \in [0, 1]$.

### 1.4 Scene-Cut Detection
Computes a weighted combination of the 64-bin luma histogram $L_1$ distance ($0.65 \cdot \frac{1}{2}\|h_t - h_{t-1}\|_1$) and frame-to-frame Mean Absolute Difference ($0.35 \cdot \mathrm{MAD}$). Cuts exceeding `scene_cut_threshold` (default `0.35`) increment the stream segment ID and isolate temporal windows.

---

## 2. Spatial Enhancement (`src/enhance/spatial_enhance.cpp`)

- **Adaptive Deblocking (`deblock`)**: In `auto` mode, activates when `blockiness > 0.12`. Applies a 4-tap boundary-straddling H.264-style loop filter across $8 \times 8$ horizontal/vertical boundaries only when $|p_0 - q_0| < \alpha$ and interior shoulder steps $|p_1 - p_0|, |q_1 - q_0| < \beta$, preserving real object edges that coincide with block lines.
- **Adaptive Denoising (`denoise`)**: In `auto` mode, activates when $\hat{\sigma} > 0.009$. Applies an edge-preserving bilateral filter ($\sigma_{\text{color}} \propto 2.8\hat{\sigma}$, $\sigma_{\text{space}} \in [1.0, 2.5]$) softly blended using gradient magnitude so micro-edges are not plasticized.
- **Adaptive Sharpening & Halo Suppression (`sharpen`)**: Computes a Gaussian detail layer $D = Y - G_\sigma * Y$, gates it with gradient-based coring (preventing noise amplification in flat zones and attenuating already high-contrast edges), and clamps output pixels to $[\min_{3\times 3}(Y) - \delta, \max_{3\times 3}(Y) + \delta]$ (`halo_overshoot`, default `0.02`) to eliminate ringing halos.

---

## 3. Edge-Directed Steering-Kernel Interpolation (`src/reconstruct/spatial.cpp`)

1. Computes the smoothed LR structure tensor $J = G_{1.0} * (\nabla Y \, \nabla Y^T)$ with eigenvalues $\lambda_1 \ge \lambda_2$, coherence $c = \frac{\lambda_1 - \lambda_2}{\lambda_1 + \lambda_2 + \epsilon}$, and normal orientation $\phi$.
2. Near coherent edges ($g = c^2 \cdot \mathrm{smoothstep}(0.004, 0.03, \sqrt{\lambda_1})$), replaces bicubic interpolation with an anisotropic Gaussian Nadaraya–Watson kernel elongated along the edge tangent ($\sigma_n = 0.38$, $\sigma_t = 0.45 + 1.35 c$).

---

## 4. Single-Frame Huber + BTV Iterative Back-Projection (`src/reconstruct/ibp.cpp`)

Models LR formation as Gaussian PSF blur followed by exact $s \times s$ box-averaging:
$$\hat{Y} = D_s(H_{\sigma} * X), \quad \sigma = (0.26 + 0.16\,t)\,s$$
where $t = \mathrm{clamp}((\text{blur\_score} - 0.25)/0.35, 0, 1)$ adapts the effective PSF width to measured source blur when `psf_sigma <= 0` (`auto`). Minimizes the regularized energy:
$$E(X) = \sum_{i} \rho_\delta\big(\hat{Y}_i - Y_i\big) + \lambda_{\text{BTV}} \sum_{|l|,|m| \le P} \alpha^{|l|+|m|} \| X - S_x^l S_y^m X \|_1$$
where $\rho_\delta$ is the Huber loss (`huber_delta = 0.050`), $\lambda_{\text{BTV}}$ scales adaptively with measured noise/blockiness, and $S_x^l S_y^m$ are spatial shift operators (Farsiu et al., 2004). Initialized from Edge-Directed interpolation and executed over overlapping LR tiles (`tile_size = 192`, `pad = 8`) in parallel via OpenMP.

---

## 5. Motion Estimation & Occlusion Gating (`src/motion/optical_flow.cpp`)

- Computes bidirectional Dense Inverse Search (DIS) optical flow $\mathbf{u}: I_t \to I_{t+k}$ and $\mathbf{v}: I_{t+k} \to I_t$.
- **Forward–Backward Consistency**: Cycle error $e_{\text{FB}}(p) = \|\mathbf{u}(p) + \mathbf{v}(p + \mathbf{u}(p))\|_2$. Pixels out-of-bounds or with $e_{\text{FB}} > \tau_{\text{fb}}$ (`0.5` LR px) are marked occluded (`occlusionMask = 0`).
- **Photometric Outlier Rejection**: Warped residual $r(p) = |I_{t+k}(p + \mathbf{u}(p)) - I_t(p)|$; pixels with $r(p) > \tau_{\text{reject}}$ (`0.12`) are rejected.
- **Confidence Weight**: $c(p) = \exp\!\left(-\frac{e_{\text{FB}}^2}{2(\tau_{\text{fb}}/2)^2}\right)\exp\!\left(-\frac{r^2}{2\sigma_{\text{photo}}^2}\right)$, truncated to $0$ below `min_confidence` (`0.15`).

---

## 6. Five-Frame Multi-Frame Super-Resolution (`src/reconstruct/mfsr.cpp`)

For reference frame $I_t$ with window $I_{t-2}, I_{t-1}, I_t, I_{t+1}, I_{t+2}$:
1. **Frame selection**: Exclude any neighbor whose scene segment differs from $I_t$ or whose mean confidence $< 0.03$.
2. **Sub-pixel alignment**: Warp each neighbor directly onto the HR grid via pixel-centered coordinates $(X+0.5)/s - 0.5 + \mathbf{u}_k$, and erode confidence masks by 1 LR pixel to prevent disocclusion edge ringing.
3. **Confidence-weighted temporal fusion**: Fuse $X_{\text{spatial}}$ (weight $1.0$) with aligned neighbors weighted by $\gamma_k c_k$ ($\gamma_k = 0.85^{|k|-1}$).
4. **Multi-frame Huber + BTV IBP**: Project $X^{(m)}$ into each valid neighbor's HR frame (`CV_16SC2` cubic remap), evaluate $D_s(H_\sigma * X_k) - Y_k$ gated by warped confidence, back-project residuals to $I_t$'s HR grid, normalize by local weight sum, add BTV gradient, and step.
5. **Observation consistency check**: If multi-frame iterations raise the reference observation RMSE above `1.08 * initial_rmse`, reject multi-frame updates and fall back to single-frame IBP.
6. **Optional temporal refinement (`temporal_refine > 0`, experimental)**: Motion-compensated blend with previous HR output within the same scene segment.

---

## Mathematical Assumptions & Known Limitations

- Assumes Lambertian brightness constancy over the 5-frame window (mild global shifts handled by DIS mean normalization).
- Assumes 8-bit 4:2:0 planar luma/chroma processing; 10-bit+ sources are converted to 8-bit on decode.
- Classical PSF + motion SR cannot synthesize textures that were completely destroyed by heavy quantization or extreme blur (`blur_score > 0.65`, `8x` scaling).
- Chroma (`U, V`) uses fast spatial interpolation rather than multi-frame IBP.
