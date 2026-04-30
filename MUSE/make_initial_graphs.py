import matplotlib.pyplot as plt

# Labels
methods = ["Pre-Unlearning", "Reproduced Extraction", "Post-Unlearning"]

# Data
aesr_09 = [0.0063, 0.0060, 0.0040]
aesr_099 = [0.0052, 0.0052, 0.0032]
rouge = [0.2992, 0.3012, 0.2984]


def make_bar_plot(values, title, filename):
    plt.figure(figsize=(6, 4))
    
    bars = plt.bar(methods, values)
    
    plt.title(title)
    plt.ylabel("Score")
    
    # Add value labels
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.4f}",
            ha='center',
            va='bottom'
        )
    
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


# Create plots
make_bar_plot(aesr_09, "A-ESR (τ = 0.9)", "aesr_09.png")
make_bar_plot(aesr_099, "A-ESR (τ = 0.99)", "aesr_099.png")
make_bar_plot(rouge, "Rouge-L Recall", "rouge.png")

plt.show()
