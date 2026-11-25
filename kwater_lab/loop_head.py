"""Small CPU cross-attention probe with one query per risk and optional tied depth."""

import numpy as np

from .train_head import sigmoid


def initialize(width, positives, training_size, seed, independent_blocks):
    rng = np.random.default_rng(seed)

    def weight(rows, cols, scale):
        return (rng.standard_normal((rows, cols)) * (scale / np.sqrt(rows))).astype(np.float32)

    codes = len(positives)
    prior = (positives.astype(np.float32) + 1) / (training_size + 2)
    params = {"queries": weight(codes, width, 0.6), "readout": weight(codes, width, 0.4),
              "bias": np.log(prior / (1 - prior))}
    block = {"query": weight(width, width, 0.8), "key": weight(width, width, 0.8),
             "value": weight(width, width, 0.45), "up": weight(width, width * 2, 0.7),
             "down": weight(width * 2, width, 0.3)}
    # Both two-pass models start from the same function. Only the sharing constraint differs.
    for index in range(independent_blocks):
        params.update({f"{index}/{name}": value.copy() for name, value in block.items()})
    return params


def forward(params, patches, block_order):
    state = patches.mean(axis=1)[:, None, :] + params["queries"]
    history = []
    for index in block_order:
        prefix = f"{index}/"
        query = state @ params[prefix + "query"]
        key = patches @ params[prefix + "key"]
        value = patches @ params[prefix + "value"]
        scores = (query @ key.swapaxes(1, 2)) * (patches.shape[-1] ** -0.5)
        attention = np.exp(scores - scores.max(axis=2, keepdims=True))
        attention /= attention.sum(axis=2, keepdims=True)
        attended = state + attention @ value
        hidden = np.maximum(attended @ params[prefix + "up"], 0)
        history.append((prefix, state, query, key, value, attention, attended, hidden))
        state = attended + hidden @ params[prefix + "down"]
    logits = (state * params["readout"]).sum(axis=2) + params["bias"]
    return logits, state, history


def _weight_gradient(inputs, gradient):
    return inputs.reshape(-1, inputs.shape[-1]).T @ gradient.reshape(-1, gradient.shape[-1])


def loss_and_gradient(params, patches, targets, block_order):
    logits, state, history = forward(params, patches, block_order)
    loss = np.mean(np.logaddexp(0, logits) - targets * logits)
    residual = (sigmoid(logits) - targets) / targets.size
    gradients = {key: np.zeros_like(w) for key, w in params.items()}
    gradients["bias"] = residual.sum(axis=0)
    gradients["readout"] = (residual[:, :, None] * state).sum(axis=0)
    delta = residual[:, :, None] * params["readout"]
    for prefix, previous, query, key, value, attention, attended, hidden in reversed(history):
        gradients[prefix + "down"] += _weight_gradient(hidden, delta)
        hidden_grad = (delta @ params[prefix + "down"].T) * (hidden > 0)
        gradients[prefix + "up"] += _weight_gradient(attended, hidden_grad)
        attended_grad = delta + hidden_grad @ params[prefix + "up"].T
        value_grad = attention.swapaxes(1, 2) @ attended_grad
        gradients[prefix + "value"] += _weight_gradient(patches, value_grad)
        attention_grad = attended_grad @ value.swapaxes(1, 2)
        score_grad = attention * (attention_grad - (attention_grad * attention).sum(axis=2, keepdims=True))
        score_grad *= patches.shape[-1] ** -0.5
        query_grad = score_grad @ key
        gradients[prefix + "key"] += _weight_gradient(patches, score_grad.swapaxes(1, 2) @ query)
        gradients[prefix + "query"] += _weight_gradient(previous, query_grad)
        delta = attended_grad + query_grad @ params[prefix + "query"].T
    gradients["queries"] = delta.sum(axis=0)
    return loss, gradients


def fit_head(patches, targets, order, seed, steps, learning_rate):
    params = initialize(patches.shape[-1], targets.sum(axis=0), len(targets), seed, len(set(order)))
    first = {key: np.zeros_like(w) for key, w in params.items()}
    second = {key: np.zeros_like(w) for key, w in params.items()}
    for step in range(1, steps + 1):
        loss, gradients = loss_and_gradient(params, patches, targets, order)
        if not np.isfinite(loss):
            raise ValueError("Non-finite loss in the recurrent visual probe")
        for key, gradient in gradients.items():
            first[key] = 0.9 * first[key] + 0.1 * gradient
            second[key] = 0.999 * second[key] + 0.001 * np.square(gradient)
            moment = first[key] / (1 - 0.9 ** step)
            variance = second[key] / (1 - 0.999 ** step)
            params[key] -= learning_rate * moment / (np.sqrt(variance) + 1e-8)
    return params
