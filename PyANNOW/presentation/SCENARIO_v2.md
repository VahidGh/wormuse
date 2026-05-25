# PyANNOW v2.0.0 — 20-Minute Speaking Scenario

> A complete script for the NAML project presentation.  
> Time targets are in **[brackets]**. Total: ~20 minutes.  
> Worm dance is running live in the right panel throughout.

---

## Before you begin

Open `http://localhost:7432/presentation/index_v2.html` in full-screen (press F).  
Wait ~5 s for the worm frame to load and animation to start.  
Confirm the worm is moving and Chopin is audible (or suppress audio if room has noise).

---

## Slide 1 — Title [0:00–0:45]

*[Advance to slide 1. Glance right at the dancing worm.]*

> "The title of this project is *The Worm Dances to Chopin* — and as you can see on the right, that is happening right now, live, in your browser.

> That is a simulation of *C. elegans* — a millimeter-long nematode with exactly 302 neurons. Every muscle contraction you see on that screen is being driven in real time by patterns extracted from Chopin's Nocturne in C# minor.

> The question I want to answer in the next 20 minutes is: how do you get from a MIDI file to a dancing worm using nothing but the methods from our NAML course?"

---

## Slide 2 — Presenter [0:45–2:00]

*[Advance to slide 2.]*

> "A quick word about why I'm interested in this.

> I've been a contributor to the OpenWorm Foundation since 2014 — it's a non-profit building the first complete computational model of *C. elegans*. My specific work there was ChannelWorm: curating patch-clamp electrophysiology data, fitting Hodgkin-Huxley models to ion channels, building an API for the worm's ion-channel database.

> That led to two published papers — one in the Philosophical Transactions of the Royal Society, one in F1000Research — and an invited talk at The Royal Society in London in 2018.

> I'm now doing my M.Sc. in HPC Engineering here at PoliMi, which is why NAML is directly relevant: I want to understand the mathematical foundations of ML, not just use the libraries.

> PyANNOW started as a way to use this real worm data as a substrate for applying every method in the NAML course."

---

## Slide 3 — Agenda [2:00–2:30]

*[Advance to slide 3. Quick.]*

> "Here's the road map. We'll cover v1 in about one minute — what we built last semester. Then five NAML blocks: the data representation, RSVD, K-means, excitability, and finally the dance. I'll end with the full NAML connections table and a few open questions."

---

## Slide 4 — v1.0.0 Recap [2:30–4:00]

*[Advance to slide 4.]*

> "In v1, the question was: *can a worm play Chopin?* — in the direction from worm to music.

> We ran a Hodgkin-Huxley simulation of the worm's 302 neurons, extracted 96 body-wall muscle signals, mapped them to piano notes, and compared the result to Chopin using onset F1.

> We went through 10 NAML steps: SVD + Procrustes, K-means, ridge regression, a JAX feed-forward network, Adam, L-BFGS. Best result was F1=0.879 with Step 9 — worm features combined with Fourier time embeddings.

> But there was a fundamental problem. [pause]

> The HH simulator, when driven by constant command input, reaches a *fixed-point attractor*. Every starting condition converges to the same periodic muscle pattern. That makes a beautiful mathematical point — but a terrible dance visualisation, because the worm always does the same thing.

> v2 turns the pipeline around: instead of worm→music, we go *music→patterns→worm dance*. We extract recurring patterns from Chopin first, then drive the muscles from those patterns."

---

## Slide 5 — Piano-Roll [4:00–5:20]

*[Advance to Part I header, then the piano-roll slide.]*

> "The starting point is the piano-roll matrix M. This is standard in music information retrieval — you take a MIDI file, and you create a binary matrix where rows are pitch bins and columns are time frames.

> Chopin's nocturne at 20-millisecond resolution gives us M ∈ ℝ^{56×11711} — 56 unique pitches, 11,711 time frames, about 234 seconds of music.

> Each entry $M_{p,t}$ is 1 if pitch $p$ is sounding at time $t$, and 0 otherwise. About 9% of entries are active at any time.

> This is the same matrix structure as in Lab01 — there we had pixels × image variants; here we have pitches × time. The linear algebra is identical."

---

## Slide 6 — 96 Muscles = 96 Piano Keys [5:20–6:20]

*[Advance to slide 6.]*

> "Now the worm side. The Boyle 2012 model from *PLoS Computational Biology* describes C. elegans body-wall muscles in four quadrants, each with 24 segments from head to tail: dorsal-left, ventral-left, dorsal-right, ventral-right.

> 4 × 24 = 96 muscles.

> And here's a beautiful coincidence: 8 octaves × 12 semitones = 96 MIDI notes — the full piano keyboard.

> So we can assign one unique pitch to each muscle: DL gets the bass register, VR gets the treble. The body wave travelling head-to-tail corresponds to pitch rising from C1 to B8.

> The phase structure: dorsal quadrants fire in phase, ventral quadrants fire 180° anti-phase. This is the biological mechanism that generates propulsive thrust — and it creates a mathematical trap that I'll show you in a few slides."

---

## Slides 7–8 — RSVD + Eckart-Young [6:20–9:00]

*[Advance to Part II header, then Eckart-Young slide.]*

> "Now the core NAML content. We want to extract the main patterns from M — the recurring musical structures in the nocturne.

> The Eckart-Young theorem tells us exactly how to do this optimally. Given any matrix M, the truncated SVD $\hat{M}_k = U_k \Sigma_k V_k^T$ is the rank-k matrix that minimises the Frobenius norm $\|M - \hat{M}_k\|_F$. It is the best possible rank-k approximation — no other basis of dimension k captures more variance.

> This is exactly what we used in Lab01 for image compression. Here we apply it to music.

> The components have clear musical interpretations: $U_k[:,i]$ is a pitch profile — which pitches co-activate in pattern $i$. $V_k[:,i]$ is the temporal envelope — when pattern $i$ is active during the piece. $\sigma_i$ is pattern energy — $\sigma_1$ alone captures 42% of the total variance."

*[Advance to RSVD slide.]*

> "For a 56 × 11,711 matrix, we use randomized SVD — the Halko 2011 algorithm from lecture 9.

> The key idea: instead of computing all singular vectors, we project M onto a random sketch of dimension k + p, run a couple of power iterations to sharpen the result, then compute SVD on the small sketch. Cost O(k·PT) instead of O(min(P,T)·PT).

> [Point to scree plot.] The scree plot shows the variance fraction per component. We choose k=12 by the 90% cumulative variance criterion — the elbow in the curve. k=12 explains 90% of the piano-roll's total energy in a fraction of a second."

---

## Slides 9–10 — K-means + Silhouette [9:00–11:30]

*[Advance to Part III header, then K-means slide.]*

> "Now we have $V_k \in \mathbb{R}^{T \times k}$ — 11,711 time frames, each described as a k-dimensional mode-coordinate vector. We want to find the *recurring musical states* — the phrases that repeat throughout the nocturne.

> This is exactly the K-means problem from lecture 10 and lab 02. Each row of $V_k$ is a point in k-dimensional space. K-means clusters these points to minimise total intra-cluster variance.

> Think of it as: Lab02 clustered MNIST digit images in PCA space. Here we cluster music frames in SVD mode space. The algorithm is identical — the interpretation is musical.

> Each cluster centroid $\mu_j$ is the average mode-coordinate fingerprint of one musical state."

*[Advance to silhouette slide.]*

> "How do we choose K? The silhouette score. For each point, we compute $s(i) = (b(i) - a(i)) / \max(a(i), b(i))$, where $a(i)$ is average intra-cluster distance and $b(i)$ is average distance to the nearest other cluster. $s$ close to 1 means tight, well-separated clusters.

> [Point to silhouette chart.] The peak is at K=8 — eight musical states optimally separate and internally coherent.

> Eight patterns in a nocturne is musically reasonable. A Chopin nocturne typically has an ABA structure with transitional passages — you'd expect 6–10 distinct harmonic states. The silhouette criterion recovered this from pure geometry, with no musical knowledge encoded."

---

## Slides 11–12 — Excitability + Least Squares [11:30–14:00]

*[Advance to Part IV header, then excitability slide.]*

> "We now have 8 musical patterns from K-means. But which of these patterns does the worm actually 'resonate with'? Which Chopin motifs align with the timescales and rhythms of *C. elegans* neural oscillation?

> We measure this with Pearson cross-correlation. The excitability of pattern $i$ is the maximum absolute Pearson r between the temporal envelope $V_p[:,i]$ and any worm neural score $V_\text{worm}[:,j]$.

> Pearson r is the right metric here because it's scale-invariant — it measures *shape* correlation, not amplitude. The worm's neural oscillations are around 0.5–2 Hz, and Chopin's nocturne is around 69 BPM, so some patterns align with worm timescales and some don't."

*[Advance to least squares slide.]*

> "Once we know the excitability ranking, we need to map each musical pattern to a 96-muscle activation pose — a concrete body shape for the worm.

> This is a least-squares problem: given the worm's neural score matrix $Z_\text{worm}$ and the muscle mode matrix $V_\text{mus}$, find the linear map $W_{nm}$ that best predicts muscle activation from neural scores.

> $W_{nm} = \text{lstsq}(Z_\text{worm}, V_\text{mus}) = Z_\text{worm}^+ V_\text{mus}$.

> We use the pseudoinverse because the system can be rank-deficient — the HH simulator's periodic attractor means the neural score columns are nearly linearly dependent. The pseudoinverse gives the minimum-norm solution and handles this gracefully.

> Each cluster average of $Z_\text{worm}$ times $W_{nm}$ gives a 96-dimensional muscle pose for that musical pattern. These 8 poses drive the worm's dance."

---

## Slides 13–14 — Body Wave + L/R Fix [14:00–16:30]

*[Advance to Part V header, then body wave slide.]*

> "The synthetic body wave function `synthMusFromVp` maps 4 piano temporal modes to locomotion signals.

> Mode 0 controls overall amplitude. Mode 1 controls the body-wave phase. Mode 2 controls D/V balance. Mode 3 controls the L/R turning offset.

> The wave equation is $\text{mus}[q,s] = A \cdot (1 + \sin(\phi(s) + \phi_q)) / 2$, where $\phi(s) = \phi_0 + s \cdot 2\pi / 24$ propagates head to tail, and $\phi_q$ sets the phase offset per quadrant.

> Now here's an interesting problem I ran into."

*[Advance to L/R problem slide.]*

> "My first version of the navigation code used the mean of dorsal-left plus ventral-left as the 'left movement' signal, and same for the right side. And the worm always went straight. Zero turning. Flat.

> After some head-scratching I proved why.

> DL body-wave amplitude is $A/2 \cdot (1 + \sin\theta)$. VL is $A/2 \cdot (1 - \sin\theta)$ — because ventral is 180° anti-phase. Their sum is $A/2$ — a constant, independent of $\theta$. The locomotion signal cancels completely.

> You can average DL and VL all you like — you will never get the body-wave phase information back, because they are designed by biology to be anti-phase.

> The fix: use only the dorsal quadrant on each side — headDL vs headDR — both dorsal, both in-phase with the body wave, with just a small bilateral offset between left and right. This gives the L/R signal without anti-phase cancellation.

> [Point to bar chart.] After the fix, the behavior distribution shows genuine variety: 26% forward, 15% forward-right, 8% forward-left, 51% halt. The halt fraction is correct — C. elegans pauses naturally, and low-energy musical passages drive low-amplitude muscle activation below the movement threshold."

---

## Slide 15 — Live Demo [16:30–17:30]

*[Advance to live demo slide. Gesture toward right panel.]*

> "And here it is, running live on the right.

> The left canvas shows the worm's body. The body wave is synthesised from Chopin's current temporal mode. The color of the worm changes with the active pattern — you might notice it shifting as the music progresses through its harmonic sections.

> The trail shows the locomotion history — forward passages appear as longer straight segments, turning passages curve.

> The right canvas shows a compressed view of the neural circuit: four command interneurons (AVA/AVD/AVB/PVC), six motor neuron classes (DA/DB/VA/VB/DD/VD), and the connections to the 96 body-wall muscles shown as S-curve bezier paths.

> At the bottom, the pattern timeline shows which of the 8 musical states is active at each moment of the piece."

---

## Slide 16 — NAML Connections [17:30–18:30]

*[Advance to NAML connections table.]*

> "Here is the full accounting of every NAML method in the pipeline.

> RSVD from lectures 6–9, guaranteed by Eckart-Young. PCA equivalence connecting $V_k$ to the course's Lab02 and Lab10 content. K-means and silhouette from lecture 10 and Lab02. Least squares and the pseudoinverse from lectures 7 and 9 and Lab03. Pearson correlation from App Stat.

> The full pipeline is a composition of these primitives — each piece comes from a NAML lecture, and together they take a MIDI file to a dancing worm in one coherent data pipeline."

---

## Slide 17 — Insights [18:30–19:15]

*[Advance to insights slide.]*

> "Four things I want to emphasise.

> First: Eckart-Young isn't just a compression theorem. Applied to music, it tells you that the top singular vectors are exactly the most energetic recurring patterns — not a heuristic, a guarantee.

> Second: K*=8 is not a parameter I chose. The silhouette criterion found it from the data. That it matches musical phrase structure is a validation that the SVD + K-means pipeline is capturing real musical structure.

> Third: the pseudoinverse handles rank deficiency gracefully — which is essential when your data-generating process (HH simulation) has low intrinsic dimensionality.

> Fourth, and most surprising to me: the anti-phase cancellation. DL+VL is identically constant — not approximately, but exactly. This is a direct consequence of how biology generates locomotion. If you compute naive averages without understanding the data-generating process, you get meaningless signals."

---

## Slide 18 — Open Questions [19:15–19:45]

*[Advance to open questions slide.]*

> "A few directions for v3.

> The most interesting to me: replacing the HH ODE solver with a PINN (lecture 27). The loss would be HH PDE residual plus music-driven data loss. This would let us steer the simulation toward music-resonant trajectories while respecting the biophysics — solving the fixed-point problem without discarding the biology.

> And the closed loop: let the worm's movement generate a new piano-roll, re-run the RSVD pipeline, update the dance. The worm and the music evolve together."

---

## Slide 19/20 — Thank You [19:45–20:00]

*[Advance to thank you slide.]*

> "To summarise: RSVD finds the musical patterns, K-means names them, Pearson tells us which ones resonate with the worm, least squares maps neural modes to muscles, and the synthetic body wave makes it all move.

> [Pause. Look at worm.] The worm is still dancing.

> Code and notebook are at github.com/vahidgh/wormuse, specifically notebook 06_chopin_patterns_worm_dance_v2. Happy to discuss any of the NAML methods or the biology — thank you."

---

## Timing Summary

| Slide | Content | Target time |
|---|---|---|
| 1 | Title + live worm intro | 0:00–0:45 |
| 2 | Presenter introduction | 0:45–2:00 |
| 3 | Agenda | 2:00–2:30 |
| 4 | v1.0.0 recap + fixed-point problem | 2:30–4:00 |
| 5–6 | Piano-roll + 96-muscle architecture | 4:00–6:20 |
| 7–8 | Eckart-Young + RSVD + scree plot | 6:20–9:00 |
| 9–10 | K-means on V_k + silhouette K*=8 | 9:00–11:30 |
| 11–12 | Pearson excitability + lstsq muscle map | 11:30–14:00 |
| 13–14 | Body wave + anti-phase L/R fix | 14:00–16:30 |
| 15 | Live demo walkthrough | 16:30–17:30 |
| 16 | NAML connections table | 17:30–18:30 |
| 17–18 | Insights + open questions | 18:30–19:45 |
| 19–20 | Thank you | 19:45–20:00 |

---

## Handling common interruptions

**"Why didn't you use PyTorch for the clustering?"**  
> "The course uses JAX for neural networks but scikit-learn for classical ML (K-means, Ridge, RandomForest). I followed that convention — sklearn's K-means is efficient, well-tested, and matches the Lab02 pattern exactly."

**"What's the F1 score for v2?"**  
> "V2 doesn't have an F1 metric — it's not predicting Chopin note onsets. The output is the visualisation. If you want a quantitative measure, the silhouette score at K*=8 and the Pearson r values for each pattern are the analogous quality metrics."

**"Why not use a deep learning model for pattern extraction?"**  
> "We could. A Transformer or CNN trained on music could extract patterns with higher F1. But the point of this project is NAML — using only the methods from the course. RSVD + K-means is interpretable, fast, and the patterns it finds are musically meaningful."

**"The worm seems to halt a lot — is that realistic?"**  
> "Yes — C. elegans naturally spends 40–60% of time in quiescence. Low-energy musical passages drive low-amplitude muscle activation. The halt behavior is a correct biological reflection of the music's dynamic range."
