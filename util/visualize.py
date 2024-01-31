import matplotlib.pyplot as plt

def plot_dataset_label_histogram(dataset, image_path):
    labels = dataset.dataset.targets
    labels = [labels[i] for i in dataset.indices]
    plt.hist(labels)
    plt.savefig(image_path)