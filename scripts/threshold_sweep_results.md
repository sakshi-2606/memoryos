# Threshold calibration sweep (F2)

Dataset: v1.1 — 6 probes, 16 facts
Similarity: true cosine = 1 - chroma_cosine_distance
Operating-point rule: highest threshold with recall=1.0 for every probe

```
 thresh | avg_prec | avg_rec | recall=1 all?
--------------------------------------------
   0.00 |    0.085 |   1.000 | yes
   0.05 |    0.090 |   1.000 | yes
   0.10 |    0.149 |   1.000 | yes
   0.15 |    0.274 |   1.000 | yes
   0.20 |    0.403 |   1.000 | yes
   0.25 |    0.653 |   1.000 | yes
   0.30 |    0.889 |   1.000 | yes
   0.35 |    1.000 |   1.000 | yes
   0.40 |    1.000 |   1.000 | yes
   0.45 |    0.833 |   0.833 | no
   0.50 |    0.833 |   0.833 | no
   0.55 |    0.667 |   0.667 | no
   0.60 |    0.333 |   0.333 | no
   0.65 |    0.167 |   0.167 | no
   0.70 |    0.167 |   0.167 | no
   0.75 |    0.000 |   0.000 | no
   0.80 |    0.000 |   0.000 | no
   0.85 |    0.000 |   0.000 | no
   0.90 |    0.000 |   0.000 | no

Per-probe separation (true cosine):

  p1: relevant=0.713 | top distractor=0.344 | margin=+0.369
  p2: relevant=0.580 | top distractor=0.290 | margin=+0.290
  p3: relevant=0.630 | top distractor=0.239 | margin=+0.391
  p4: relevant=0.507 | top distractor=0.296 | margin=+0.211
  p5: relevant=0.441 | top distractor=0.239 | margin=+0.202
  p6: relevant=0.591 | top distractor=0.153 | margin=+0.437

RECOMMENDED THRESHOLD: 0.40 (avg precision 1.000, recall 1.0)
```
