# C3 P2 Seed824 Scaling Analysis

## Scope and gate

This is the required seed824-only gate: 18 new P2 runs plus six SHA-verified P1 100-image reuses. It is not a multi-seed conclusion.

### NEU-DET

- V-PEFT retention @10/50/100/500: 86.77% / 80.07% / 99.62% / 98.68%.
- V-PEFT − Full-SFT ΔmAP50-95: -0.0161 / -0.0557 / -0.0012 / -0.0052.
- Closest to Full-SFT: 100 images; largest gap: 50 images.
- Frozen Backbone is closest to the best method at 10 images; rankings: 10=Frozen Backbone > Full-SFT > V-PEFT; 50=Full-SFT > Frozen Backbone > V-PEFT; 100=Full-SFT > V-PEFT > Frozen Backbone; 500=Full-SFT > V-PEFT > Frozen Backbone.
- V-PEFT parameter reduction @10/50/100/500: 76.32% / 76.32% / 76.32% / 76.32%.
- V-PEFT memory saving @10/50/100/500: +1.53% / +1.53% / +1.16% / +1.16%.
- V-PEFT training-time change @10/50/100/500: +14.34% / +13.32% / +13.00% / +4.93%.
- Empirical Full/V-PEFT crossover region(s): none observed.

### DeepPCB

- V-PEFT retention @10/50/100/500: 76.56% / 63.87% / 80.62% / 93.78%.
- V-PEFT − Full-SFT ΔmAP50-95: -0.0584 / -0.2015 / -0.1230 / -0.0435.
- Closest to Full-SFT: 500 images; largest gap: 50 images.
- Frozen Backbone is closest to the best method at 10 images; rankings: 10=Full-SFT > Frozen Backbone > V-PEFT; 50=Full-SFT > Frozen Backbone > V-PEFT; 100=Full-SFT > V-PEFT > Frozen Backbone; 500=Full-SFT > V-PEFT > Frozen Backbone.
- V-PEFT parameter reduction @10/50/100/500: 76.32% / 76.32% / 76.32% / 76.32%.
- V-PEFT memory saving @10/50/100/500: +1.14% / +1.52% / +1.15% / +1.16%.
- V-PEFT training-time change @10/50/100/500: +19.72% / +12.39% / +11.69% / +7.53%.
- Empirical Full/V-PEFT crossover region(s): none observed.

## Required questions

1. **Where is V-PEFT closest/largest-gap?** NEU: closest=100, largest=50; DeepPCB: closest=500, largest=50.
2. **Do dataset trends agree?** NEU retention=86.77%/80.07%/99.62%/98.68%; DeepPCB=76.56%/63.87%/80.62%/93.78%. Agreement is assessed from these measured sequences, not assumed from P1.
3. **Is Frozen better at extremely low data?** At 10 images, NEU ranking is Frozen Backbone > Full-SFT > V-PEFT; DeepPCB ranking is Full-SFT > Frozen Backbone > V-PEFT. This directly answers the seed824 observation without generalizing beyond one seed.
4. **Does parameter efficiency persist?** Yes structurally: the V-PEFT trainable-parameter reduction is 76.32%/76.32%/76.32%/76.32% on NEU and 76.32%/76.32%/76.32%/76.32% on DeepPCB.
5. **Does P1-100 lie on the scaling trend?** The audited 100-image retentions are NEU 99.62% and DeepPCB 80.62%; their position relative to the 10/50/500 points is visible in the measured sequences above. No monotonicity is imposed.

## Measured dataset differences

All scales cover 6/6 classes. Measured object densities (objects/image) are NEU 10=3.70, 100=2.31, 500=3.13; DeepPCB 10=9.90, 100=7.08, 500=7.80. These statistics establish distributional differences but do not prove a mechanism for accuracy trends.

## Multi-seed decision

`MULTISEED_READY=YES` because all 24 seed824 cells are finite and traceable, nesting/reuse audits pass, and the observed curves contain sample-size-dependent differences worth estimating across seeds. This decision authorizes a later plan only; seed825/826 were not run here.
