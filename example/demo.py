from code_bert_score import score

with open("hyps.txt") as f:
    cands = [line.strip() for line in f]

with open("refs.txt") as f:
    refs = [line.strip() for line in f]

(P, R, F1, F3), hashname = score(cands, refs, lang="en", return_hash=True)
print(f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F1={F1.mean().item():.6f} F3={F3.mean().item():.6f}")
