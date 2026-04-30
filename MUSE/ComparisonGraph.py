import matplotlib.pyplot as plt
import numpy as np

methods = ["Pre-Unlearning Model", "Extraction Method", "Post-Unlearning Model"]

# Exact unlearning (My reproduced results)
exact = [0.4798, 0.6119, 0.2992]

# DP-Adam results (my final run)
dp = [0.2255, 0.2247, 0.2248]

x = np.arange(len(methods))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(x - width/2, exact, width, label="Exact Unlearning", edgecolor='none', linewidth=0)
plt.bar(
    x + width/2, 
    dp,
    width, 
    label="DP-Adam (σ=1.0, C=1.0, δ=1e−5, ε≈0.026)",
    edgecolor='none',
     linewidth=0
)

plt.xticks(x, methods)
plt.ylabel("Rouge-L Recall")
plt.title("Extraction under Exact vs Approximate Unlearning")
plt.legend(loc="upper right")
plt.grid(axis='y')

plt.tight_layout()
plt.savefig("ExactVApprox_graph.png", dpi=300)
plt.show()
