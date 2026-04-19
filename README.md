# Sufficient Cause Disambiguation (SCD)

**Which reason did the model actually rely on?**

When a language model classifies a document, multiple features often independently justify the label. Standard attribution methods tell you which tokens were important. They cannot tell you which one the model was actually relying on for that specific prediction. SCD solves this. It identifies the operative sufficient cause by combining LDA on hidden states, input-gradient attribution, and causal activation patching applied to the output weight direction.

The framework is built on Rothman's (1976) epidemiological Sufficient Cause Model and applied to Llama-3-8B-Instruct on WOS-46985. The central finding is that the subclass representational subspace and the output decision axis are causally independent: removing everything the model knows about subclass identity from its hidden state leaves the prediction unchanged, while flipping the prediction moves the representation into a different CS subclass, not into Medical. Two fully independent subspaces.

Course project for CS 7180 Actionable Interpretability, Northeastern University, Spring 2026. Advisor: Bryan Wallace.

---

## Results

| Finding | Result |
|---|---|
| Subclass accuracy, 3 LDA dims vs 4,096 raw | **99.6% vs 57.5%** (448x compression) |
| Flip rate patching lm_diff_n at alpha=2 | **100%** (n=38 ambiguous CS_sub7) |
| Flip rate patching any LDA direction at any alpha | **0%** |
| Delta-logit per unit alpha | **2.043** (exactly linear, confirmed experimentally) |
| Demographics share of ambiguous Fit predictions | **62%** on identical content, different candidate name |

---

## Notebooks

| Notebook | Drive cache | What it does |
|---|---|---|
| `SCD_Final_Notebook_final.ipynb` | `SCD_Final.zip` | Main pipeline: LDA geometry, gradient attribution, causal patching, full population results. Last cell runs the LDA axis intervention. |
| `SCD_LayerToken_Sweep_final.ipynb` | `SCD_LayerStrat.zip` | Sweeps 70 layer x token strategy combinations and justifies layer 29 last token as the extraction point. |
| `SCD_RealWorld_final.ipynb` | `SCD_RealWorld.zip` | Applies the same pipeline without modification to resume screening across skills, credentials, and demographics tracks. |
| `SCD_LDA_Flip.ipynb` | `SCD_Final.zip` | Tests whether patching along LDA axes, including the principled centroid-to-centroid direction, can flip predictions. It cannot. |

---

## Setup

**You need:** Google Colab with an A100 GPU and a HuggingFace account with approved access to `meta-llama/Meta-Llama-3-8B-Instruct`.

**Step 1.** Add your HuggingFace token as a Colab secret named `HF_TOKEN` (left sidebar, key icon).

**Step 2.** Download `SCD_cache.zip` from [Google Drive](https://drive.google.com/file/d/1U0s4PGrEeTJKEIthnmVSA4KHRzSUPGHT/view?usp=sharing). Unzip it to get three inner zips:

```
SCD_cache.zip
├── SCD_Final.zip
├── SCD_LayerStrat.zip
└── SCD_RealWorld.zip
```

Upload each to Google Drive and unzip in place. Your Drive should look like this:

```
MyDrive/
├── SCD_Final/
│   ├── data/
│   ├── checkpoints/
│   ├── figures/
│   └── results/
├── SCD_LayerStrat/
│   ├── checkpoints/
│   └── figures/
└── SCD_RealWorld/
    ├── checkpoints/
    └── figures/
```

**Step 3.** Open any notebook in Colab, mount Drive, run top to bottom. Every experiment checks for a cached `.pkl` before running inference, so notebooks with the cache skip the expensive model calls automatically.

Without the cache, the full main pipeline takes around 2 to 3 hours on an A100.

---

## Dataset and Model

- **Dataset:** [`HDLTex/web_of_science`](https://huggingface.co/datasets/HDLTex/web_of_science) on HuggingFace. 46,985 academic abstracts with three-level hierarchical labels. Downloaded automatically on first run.
- **Model:** [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). Requires HuggingFace access approval.

---

## Method

**Output direction**

For CS vs Medical classification, the logit gap is h dot (W[CS] minus W[Medical]). We normalise this:

```
lm_diff_n  =  (W[CS] - W[Medical])  /  ||W[CS] - W[Medical]||
```

Every prediction reduces to h dot lm_diff_n. One fixed direction computed from the weight matrix, used unchanged throughout every experiment.

**Causal patch**

```
h_new  =  h  -  alpha x (h . lm_diff_n) x lm_diff_n
```

Removes the component of the hidden state pointing toward CS. Implemented as a PyTorch forward hook on layer 29. The delta-logit is exactly alpha times 2.043, guaranteed by linearity.

**Gradient attribution**

```
attr(token_i)  =  grad_i . emb_i
```

where `grad_i = d(logit_CS - logit_Medical) / d(emb_i)`. Gradient sensitivity weighted by actual activation magnitude. Derived from the first-order Taylor expansion around the zero embedding baseline.

**Extraction point**

Layer 29, last token. Selected from a sweep of 70 combinations (10 layers x 7 token strategies). Two critical observations from the sweep: layer 0 last token gives exactly random accuracy (0.333), confirming the model must process the document first. Separator tokens give 1.000 LDA accuracy across all deep layers but produce 0% flip rate under patching. Perfect representational content, zero causal relevance. This is why extraction point selection requires both a representational criterion and a causal one.

---

## What We Found

Subclass identity concentrates into 3 LDA dimensions at 99.6% accuracy from 4,096, with one scalar sufficient to achieve the same result. The output direction flips every CS prediction given enough alpha, with a perfectly linear logit shift. Patching along LDA axes, including all three combined and the direction that points from the CS_sub7 centroid toward the Medical_sub5 centroid in representational space, produces zero flip at any alpha. The two subspaces are causally independent.

In resume screening, name-swapped resumes receive nearly double the Fit prediction rate on identical content (+0.692 logit points) and flip faster under patching (95.8% vs 80% at alpha=0.5). The name signal produces more concentrated, more fragile predictions than genuine skills evidence does. 62% of the model's most uncertain Fit predictions come from the track that differs only in candidate name.

---

## References

- Rothman, K. J. (1976). Causes. *American Journal of Epidemiology*, 104(6), 587-592.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML 2017*.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?". *KDD 2016*.
- Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). Interpretability in the wild. *ICLR 2023*.
- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT. *NeurIPS 35*.
- Geiger, A., Lu, H., Icard, T., & Potts, C. (2021). Causal abstractions of neural networks. *NeurIPS 34*.
- Belinkov, Y., & Glass, J. (2019). Analysis methods in neural language processing. *TACL*, 7, 49-72.
- Wilson, K., & Caliskan, A. (2024). Gender, race, and intersectional bias in resume screening. *AIES 2024*.
- Dubey, A., et al. (2024). The Llama 3 herd of models. arXiv:2407.21783.
