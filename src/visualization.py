import matplotlib.pyplot as plt


def plot_training_loss(epoch_loss_main, epoch_loss_reduced, output_path="training_loss_comparison.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_loss_main, label="GPT2")
    plt.plot(epoch_loss_reduced, label="GPT2 (reduced)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()


def plot_validation_metrics(
    scenario_labels,
    loss_values,
    acc_values,
    loss_output_path="validation_loss_comparison.png",
    acc_output_path="validation_accuracy_comparison.png",
):
    plt.figure(figsize=(9, 5))
    plt.bar(scenario_labels, loss_values, color=["#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e"])
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss by Scenario")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(loss_output_path, dpi=150)
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.bar(scenario_labels, acc_values, color=["#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e"])
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy by Scenario")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(acc_output_path, dpi=150)
    plt.show()
