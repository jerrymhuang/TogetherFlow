# Limitations: planar representation

The present model represents agents by their positions and heading angles in a two-dimensional plane. Its recovery and calibration results therefore establish inferential performance within a planar generative model, rather than validity for the full range of behavior supported by an immersive room. In particular, the model does not distinguish gaze from locomotor heading or represent vertical movement and responses to elevated targets. These distinctions may matter when participants explore spatially distributed content or orient toward other people without changing their direction of travel [A]. Omitted spatial behavior could affect the interpretation of inferred social influence or directional noise, although such effects have not been demonstrated by the present simulations.

A practical extension would first retain floor-constrained locomotion while incorporating three-dimensional gaze and beacon elevation. A subsequent full spatial model would require three-dimensional positions and orientations, spatial interaction and visibility rules, and corresponding observation features. Such an extension would need to be retrained and evaluated rather than assumed to inherit the present recovery and calibration performance. Validation should include recovery of the planar case under restricted motion, simulation-based calibration in the spatial model, and a comparison of inference from full spatial trajectories with inference from their planar projections. This staged approach would allow the consequences of the 2D approximation to be measured before making claims about unrestricted 3D behavior.

---

## Citation guidance — remove before manuscript submission

The bracketed letters are citation placeholders, not references to identified papers.

| Marker | Literature to include | Purpose |
|---|---|---|
| **[A]** | Empirical work on gaze–heading coordination, head/body orientation, or navigation toward spatially distributed targets in VR. Prefer a study resembling the apparatus and task. | Support the relevance of separating gaze, orientation, and movement; not evidence that this particular dataset is biased. |
| **[B]** | Methodological work on model misspecification or simulator discrepancy in Bayesian/simulation-based inference. If available, add a directly relevant study of projecting 3D movement into 2D. | Support the concern that omitted mechanisms can affect parameter interpretation. A generic discrepancy paper cannot establish the size or direction of the effect here. |
| **[C]** | A spatial gaze/navigation model or an empirical study jointly measuring planar locomotion and 3D gaze. | Optional precedent for the staged extension. If this is solely our proposed design, remove the marker. Technical rotation references belong in the appendix if implementation details are retained there. |
| **[D]** | A methodological reference for simulation-based calibration; reuse the paper's existing SBC citation if appropriate. | Support the proposed inference validation. SBC does not establish behavioral realism or agreement with human data. |

Refer to the paper's Methods/model-equation section for the current model's state variables. The geometric statements about projected speed and horizontal distance do not require citations on their own. The proposed extension and validation experiments are future work, not completed results. Prioritize [A], [B], and the existing SBC reference; include [C] only if a specific precedent is used.
