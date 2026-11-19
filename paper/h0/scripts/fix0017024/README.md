# Corrected runs of O5 EPL_shear-7777-0017024 (2026-10-27)

The two GW images of this system have nearly equal SNRs (optimal 19.9 and 20.0) and magnifications
(mu = 25.2 at a minimum, -26.7 at a saddle point). Its joint parameter estimation (results/pe/O5) was
injected with image A = the saddle point (mu = -26.7), but the GW+EM input file (inputs/gwem) labels the
minimum as A (em.gw_a = 1, em.gw_b = 2), from a later replay of the catalog in which the order of the
measured SNRs of the two images differs. The two-stage GW+EM runs therefore compared the GW posterior of
one image with the lens prediction for the other (sign of ln mu_B/mu_A, 3% in D_A, and Morse indices
(1/2,0)/(1,1/2) against the predicted (0,1/2)), giving h = 0.70 (0.56-0.90) and log evidence -87.
It is the only one of the 119 systems whose labels differ (checked: D_A, ln mu_B/mu_A, |delay| and the
Morse-index difference of every PE injection against the GW+EM input).

Fix: inputs/O5_EPL_shear-7777-0017024.json = the original input with em.gw_a and em.gw_b swapped, so
that the labels agree with the GW injection. Every current two-stage configuration of the system was rerun
with it on the GPU server (h0paper/fix0017024/run_fix.sh, same code and environment as the original runs):
Paper I  gwem_posthoc_sn (free:0:same, lcdm:0.025:EPL_shear, free:0.025:EPL_shear, ast 0.03, ast 0.1),
         gwem_posthoc_xpos_sn (0, 0.03, 0.1), gwem_posthoc_asg_sn (free:0:same, lcdm:0.025:EPL_shear);
Paper II gwem_posthoc_h0_sn, h0_ki (drawn kappa_ext), h0_nod (delay only), gwem_posthoc_h0 (flat source prior).
The full (strain + EM) runs inject the waveforms from the GW+EM input and are self-consistent.

Files read by p2lib.py, p2_numbers.py and sample.py on top of the archived ones:
h_fix.npz (h samples, keys dir|event|h and the injected kappa_ext dir|event|k), all_fix.jsonl
(summaries), posthoc/<batch>/<config>/*_summary.json (all corrected summaries; p2_numbers.py reads gwem_posthoc_sn/lcdm_k0.025_EPL_shear_sn for Appendix C), inputs/ (corrected input).

The GPU runs used a version of the input in which truth.mu_a and truth.mu_b were still in the old order
(swapped in the file above since). They enter only the truth-dependent fields of the summaries
(true_mu_a/b, pit, rel_err, covered90), which collect_fix.sh swaps and recomputes from the posteriors.
The full (strain + EM) runs of this system inject image A = the minimum, so their GW data differ from those
of the joint parameter estimation; the system is therefore excluded from the full versus two-stage
comparison (Paper II Appendix C), like the ET runs whose noise realization could not be reproduced.

Added 2026-10-27 (for Paper I): the flat-source-prior fit gwem_posthoc/free_k0_EPL_shear (no --source-norm), which
enters the "without the normalization" sentence of Paper I Sec. V B, was rerun as well (results/gwem_posthoc_flat on
the GPU server; posthoc/gwem_posthoc_flat here), with the corrected input, so its summary needs no post-processing.
