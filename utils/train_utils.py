import torch as th


def compute_normalized_entropy(probabilities):

    # Calculate entropy
    probabilities = th.clip(probabilities, 1e-8, 1.0)
    # print(th.min(probabilities, dim=0)[0].numpy().round(3), 
    #       th.max(probabilities, dim=0)[0].numpy().round(3), 
    #       th.mean(probabilities, dim=0).numpy().round(3), 
    #       th.quantile(probabilities, q=0.25, dim=0).numpy().round(3),
    #       th.quantile(probabilities, q=0.50, dim=0).numpy().round(3),
    #       th.quantile(probabilities, q=0.75, dim=0).numpy().round(3))
    log_probabilities = th.log(probabilities)
    if not th.all(th.isfinite(log_probabilities)):
        print("error on log probabilities.")
        exit()
    entropy = -th.sum(probabilities * log_probabilities, dim=-1)

    # Determine the number of classes from logits
    num_classes = probabilities.shape[-1]

    # Normalize entropy so that max entropy (uniform distribution) is 1
    normalized_entropy = entropy / th.log(th.tensor(float(num_classes)))
    # if not th.all(th.isfinite(normalized_entropy)):
    #     print("error on normalized entropy.")

    return normalized_entropy