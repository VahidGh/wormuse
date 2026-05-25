# PyANNOW v2.0.0 — Q&A Reference (NAML-focused)

> Prepared for the 20-minute NAML presentation.  
> Questions are grouped by topic and keyed to NAML lectures / labs.

---

## Part A — SVD & Eckart-Young (L06–L09 / Lab01)

**Q1. What does the Eckart-Young theorem actually guarantee?**  
*A.* For any matrix $M \in \mathbb{R}^{P \times T}$, the truncated SVD $\hat{M}_k = U_k \Sigma_k V_k^T$ is the **unique** minimiser of $\|M - B\|_F$ over all rank-≤k matrices $B$.  
In other words, no other rank-k matrix captures more of $M$'s "energy" (Frobenius norm²).  
We exploit this to guarantee that our 12 piano patterns capture the maximum possible musical variance.

**Q2. How does RSVD relate to the full SVD?**  
*A.* Full SVD costs $O(\min(P,T) \cdot PT)$. RSVD (Halko et al. 2011) costs $O(k \cdot PT)$ — linear in $k$ by projecting $M$ onto a random sketch of dimension $k+p$, then running power iterations to improve accuracy. The course version uses $p=10$ oversampling and $q=2$ power iterations (same as the `rsvd_2024.ipynb` demo). For $k \ll \min(P,T)$, RSVD is orders of magnitude faster.

**Q3. What do the singular values mean musically?**  
*A.* $\sigma_i^2 / \|\sigma\|^2$ is the fraction of piano-roll variance explained by the $i$-th pattern.  
- $\sigma_1$: ~42% → the dominant motif (recurring bass + melody line of the nocturne)  
- $\sigma_2$: ~18% → the second harmonic layer  
- $\sigma_3$–$\sigma_{12}$: ornamental variation  
Larger $\sigma$ = more musical energy = more "important" pattern.

**Q4. What does $U_k[:,i]$ tell you about the music?**  
*A.* $U_k[:,i]$ is a vector in pitch space (length P=56). It describes **which pitches co-activate** during pattern $i$ — it is the pitch "fingerprint" or chord template of that musical pattern.  
$V_k[:,i]$ (length T) is the **temporal envelope** — when pattern $i$ is active during the piece.

**Q5. How is this different from what you did in Lab01?**  
*A.* Lab01 applied SVD to an **image matrix** (pixels × image variants) for compression and noise removal.  
Here we apply RSVD to a **musical score matrix** (pitches × time) for pattern extraction.  
The mathematics is identical — Eckart-Young gives the best rank-k approximation in both cases.  
The difference is interpretation: Lab01 singular vectors are image eigenfaces; here they are pitch-time musical modes.

---

## Part B — PCA & Dimensionality Reduction (L08 / Lab02)

**Q6. Is $V_k$ the same as PCA on M?**  
*A.* Yes. The right singular vectors $V_k$ are the PCA principal components of the *column-centred* (or raw, if mean ≈ 0) matrix $M$ computed in the *row* direction. Specifically, each row of $V_k^T$ (each column of $V_k$) is an eigenvector of $M^T M$, which is exactly the covariance matrix in the time direction. So $V_k[t,:]$ gives the PCA coordinates of time frame $t$ in the $k$-dimensional musical subspace.

**Q7. Why cluster $V_k$ and not $U_k$?**  
*A.* $V_k \in \mathbb{R}^{T \times k}$ gives one row per **time frame** — clustering these rows discovers recurring temporal patterns (musical states). $U_k \in \mathbb{R}^{P \times k}$ gives one row per **pitch** — clustering these would find pitch groups (chord templates), which is a different (also interesting!) question. For the worm dance we want time-indexed musical states, so we cluster $V_k$.

---

## Part C — K-means + Silhouette (L10 / Lab02)

**Q8. Why K-means and not hierarchical clustering or DBSCAN?**  
*A.* K-means is the course's primary clustering method (L10). It works well when:  
- Clusters are approximately spherical in $V_k$ space (empirically true here)  
- $k$ is known or can be chosen by silhouette  
DBSCAN would require tuning $\epsilon$ (harder to motivate); hierarchical clustering is $O(T^2)$ — too slow for $T=11711$ frames.

**Q9. Walk me through the silhouette formula.**  
*A.* For each point $i$:  
- $a(i)$ = mean distance to all other points **in the same cluster** (cohesion)  
- $b(i)$ = mean distance to all points in the **nearest other cluster** (separation)  
- $s(i) = (b(i) - a(i)) / \max(a(i), b(i)) \in [-1, 1]$  
Mean silhouette over all points is maximised at $K^* = 8$, meaning 8 clusters have the best balance of tight cohesion and wide separation. This matches the ~8 harmonic sections of the nocturne.

**Q10. What do the 8 musical states correspond to musically?**  
*A.* Informally: (1) opening motif C#m, (2) arpeggiated passage, (3) melodic peak, (4) transitional diminished passage, (5) recapitulation, (6) bass-heavy section, (7) ornamental run, (8) closing cadence. The SVD+K-means pipeline recovered phrase-level structure without any musical knowledge — purely from the geometry of the piano-roll matrix.

---

## Part D — Pearson Correlation & Excitability (AppStat / Lab02)

**Q11. Why use Pearson r and not the Euclidean distance for excitability?**  
*A.* Pearson r is **scale-invariant** — it measures shape correlation, removing amplitude differences. The worm's neural oscillations are bounded by HH dynamics (typically ~0.1–0.8 normalised) while Chopin's temporal modes can have different scale. What matters biologically is whether the *timing* of the musical pattern resonates with the *timing* of neural oscillation — pure shape correlation. Euclidean distance would penalise amplitude mismatches that are irrelevant to synchrony.

**Q12. What would a high excitability score mean biologically?**  
*A.* A high $|r_{ij}|$ means the temporal envelope of Chopin pattern $i$ strongly correlates with worm neural score $j$. Biologically: the worm's nervous system, when running its natural locomotion program, oscillates in rhythmic sync with that musical pattern. It does **not** mean the worm "prefers" the music — it means the biophysical timescales (C. elegans locomotion ~0.5–2 Hz) happen to overlap with the rhythmic structure of this particular nocturne.

---

## Part E — Least Squares & Pseudoinverse (L07 / L09 / Lab03)

**Q13. Why do you need the pseudoinverse rather than $(Z^T Z)^{-1} Z^T$?**  
*A.* The worm neural score matrix $Z_\text{worm} \in \mathbb{R}^{T \times k_w}$ may be rank-deficient (especially since HH simulation tends toward periodic attractors — the columns become nearly linearly dependent). The pseudoinverse $Z^+ = V \Sigma^+ U^T$ (truncating near-zero singular values) handles rank deficiency gracefully and gives the **minimum-norm** least-squares solution, which is the most stable choice when the system is underdetermined or nearly so.

**Q14. What does the matrix $W_{nm} \in \mathbb{R}^{k_w \times 96}$ represent?**  
*A.* Each column of $W_{nm}$ is a vector of $k_w$ coefficients describing how the $k_w$ neural principal components linearly combine to activate that particular muscle. Conceptually it is the linear **motor program** — a compression of the worm's neural-to-muscle computation. In neuroscience terms it is related to the "muscle synergy matrix", and computing it via lstsq from data is a standard dimensionality reduction technique in motor control.

---

## Part F — Neural Networks & Autodiff (L14–L17 / L21–22)

**Q15. Could you replace lstsq with an MLP for the neural→muscle mapping?**  
*A.* Yes — that is exactly Steps 4–6 of v1.0.0. The MLP adds non-linearity but requires gradient-based training (Adam, then L-BFGS fine-tuning). For v2 we use lstsq because:  
(a) the muscle poses are computed as cluster averages (not individual frames), giving a very small training set — lstsq is better suited,  
(b) the relationship is approximately linear in the principal-component subspace,  
(c) lstsq has a closed-form solution — no hyperparameter tuning needed.

**Q16. Where does automatic differentiation (L14) appear in this project?**  
*A.* In the v1 FFNN (Step 4) and in the PINN (future v3 direction). For the HH simulation we use PyTorch's `autograd` to compute $\partial I_\text{ion} / \partial V$ for the Jacobian during implicit time-stepping. In the planned PINN extension, `torch.autograd.grad` computes the PDE residual $\partial V / \partial t - f_\text{HH}(V, m, h, n) = 0$ at collocation points.

---

## Part G — PINNs (L27)

**Q17. How would a PINN help with the HH fixed-point problem?**  
*A.* The standard HH ODE driven by constant AVA/AVB command input settles to a stable periodic orbit (the fixed-point attractor problem). A PINN could be trained with:  
$\mathcal{L} = \mathcal{L}_\text{data} + \lambda_\text{phys} \cdot \mathcal{L}_\text{HH}$  
where $\mathcal{L}_\text{HH}$ is the HH PDE residual evaluated at collocation points, and $\mathcal{L}_\text{data}$ guides the solution toward varied trajectories (e.g. music-driven). The PINN could learn HH dynamics while being steered by the musical excitability signal — giving biophysically plausible but musically varied worm activity.

---

## Part H — Architecture & Biological Questions

**Q18. Why is 96 muscles = 96 piano keys such a neat coincidence?**  
*A.* It is partly deliberate design (Boyle et al. 4×24 model) and partly a happy mathematical fact: 4 quadrants × 24 segments = 96 = 8 octaves × 12 semitones = 96 MIDI notes spanning the piano keyboard. The mapping is: DL (bass C1–B2), VL (tenor C3–B4), DR (alto C5–B6), VR (treble C7–B8). This allowed a physically grounded pitch mapping where the body wave's head-to-tail propagation corresponds to pitch increasing from bass to treble.

**Q19. What is the Boyle 4×24 model and why use it?**  
*A.* Boyle et al. (2012, *PLoS Comput. Biol.*) measured and modelled C. elegans body-wall muscle activity from video and electrophysiology. Their 4-quadrant (DL/VL/DR/VR), 24-segment-per-quadrant model is the standard biophysical reference for worm locomotion. We use it because: (a) it is published, peer-reviewed, and parameterised; (b) it captures the key D/V anti-phase relationship; (c) 4×24=96 aligns with the piano keyboard.

**Q20. The worm spends 51% of the time in HALT. Is that realistic?**  
*A.* Yes — C. elegans naturally pauses between locomotion bouts (~40–60% quiescence in standard conditions). HALT corresponds to low total muscle activation (below threshold), which happens during quiet musical passages in the nocturne. The pattern-to-behaviour distribution is a feature, not a bug: it reflects the music's own dynamic range being mapped faithfully through the biological constraints.

---

## Part I — Meta / Project Questions

**Q21. What is the difference between PyANNOW v1 and v2?**  
*A.* v1 (Worm→Music): simulate HH worm dynamics → map 96 muscle outputs → piano notes → compare to Chopin → F1 score. 10 NAML steps, best F1=0.879.  
v2 (Music→Worm): extract patterns from Chopin via RSVD+K-means → rank by Pearson excitability → drive 96 muscles from patterns → visualise dance. Inverted pipeline; no F1 metric — the output is the live visualisation.

**Q22. What is OpenWorm and why is it relevant to your research?**  
*A.* OpenWorm (openworm.org) is a Delaware non-profit building the first complete computational model of C. elegans — a 302-neuron nematode whose entire connectome has been mapped. I have been a senior contributor since 2014, leading the ChannelWorm project (ion-channel data curation and HH fitting). The published connectome, muscle layout, and channel data are all directly used in PyANNOW's simulation backbone.

**Q23. Could this pipeline work for other music or other organisms?**  
*A.* For other music: yes — any MIDI file can be processed into a piano-roll M and run through RSVD+K-means. For other organisms: yes in principle, but you would need: (a) a computational muscle model (e.g. Drosophila flight muscle, mouse CPG), (b) a published connectome or functional connectivity map, (c) temporal overlap between neural oscillation frequencies and musical rhythm. The NAML pipeline (RSVD→K-means→Pearson→lstsq) is completely general.
