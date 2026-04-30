import matplotlib.pyplot as plt

methods = ["Pre-Unlearning Model", "DP-Adam Extraction", "Post-Unlearning"]
rouge = [0.22552827456796937, 0.22474863054124058, 0.22475395353857336]

plt.figure(figsize=(7, 5))
bars = plt.bar(methods, rouge)

plt.ylabel("Average Rouge-L Recall")
plt.title("DP-Adam Extraction Results on MUSE")
plt.ylim(0.22, 0.23)

for bar, value in zip(bars, rouge):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.0001,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("dp_adam_rouge_zoomed.png", dpi=300)
plt.show()
