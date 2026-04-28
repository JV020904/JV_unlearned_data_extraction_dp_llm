import matplotlib.pyplot as plt

methods = ["Pre-Unlearning", "Our Extraction", "Post-Unlearning"]
rouge = [0.22552827456796937, 0.22474863054124058, 0.22475395353857336]

plt.figure(figsize=(7, 5))
plt.bar(methods, rouge)
plt.ylabel("Average Rouge-L Recall")
plt.title("DP-Adam Extraction Results on MUSE")
plt.ylim(0, 0.35)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("dp_adam_rouge_results.png", dpi=300)
plt.show()
