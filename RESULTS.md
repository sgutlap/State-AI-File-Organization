# StateAI V2 — URTC Evaluation Results (summary)

## StateAI V2 (our model — MiniLM dual-encoder, virtual-none abstention, near-zero cost)

| Track | Setting | action_acc | macro_F1 | abstain_F1 | MRR |
|---|---|---:|---:|---:|---:|
| T1 | Owner lock (419 tasks) | 0.888 | 0.881 | 0.902 | 0.755 |
| T2 | Transfer / unseen taxonomy (571 tasks) | 0.452 | 0.447 | — | 0.614 |
| T3 | OOD agent systems (270 tasks) | 0.885 | 0.901 | 0.810 | 0.725 |

## Compare

| Backend | T1 | T3 |
|---|---:|---:|
| Claude (frontier) | 0.952 | 0.993 |
| Codex | 0.895 | 0.963 |
| agy (gemini) | 0.931 | — |
| **StateAI V2** | **0.888** | **0.885** |
| Ollama 27B (local) | 0.845 | 0.793 |
| dense_zeroshot floor | 0.539 | 0.570 |

## Takeaways
- StateAI V2 is **competitive with frontier agents on T1/T3** at a fraction of the cost/latency (local MiniLM,)
- Frontier agents still lead (McNemar significant, p<0.05); the gap is narrow, not a blowout.
- T2 (true cross-taxonomy transfer) is the hard setting — V2 0.452 beats the dense floor 0.380 but transfer remains the open problem.
