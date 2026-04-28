import matplotlib.pyplot as plt
import numpy as np

methods = ["Pre", "Ours", "Post"]

# Exact unlearning (your reproduced results)
exact = [0.4798, 0.6119, 0.2992]

# DP-Adam results (your final run)
dp = [0.2255, 0.2247, 0.2248]

x = np.arange(len(methods))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(x - width/2, exact, width, label="Exact Unlearning")
plt.bar(x + width/2, dp, width, label="DP-Adam")

plt.xticks(x, methods)
plt.ylabel("Rouge-L Recall")
plt.title("Exact vs Approximate Unlearning (DP-Adam)")
plt.legend()
plt.grid(axis='y')

plt.tight_layout()
plt.savefig("comparison_graph.png", dpi=300)
plt.show()
