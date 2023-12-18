"""
This script extracts a CIFAR10 training subset.
- 100 samples from label 0
- 50 samples from label 1
- 10 samples from the remaining labels.

This procedure presents a few-shot and class-imbalanced training set.
"""
import os
from torchvision.datasets import CIFAR10


if __name__ == '__main__':
    for label in range(10):
        os.makedirs(f'./cifar10-train/{label}', exist_ok=True)
    data = CIFAR10('./cifar10-train', download=True, train=True)

    # save images according to the number specified in counter_goal
    counter_goal = [100, 50] + [10] * 8
    counter = [0] * 10
    for img, label in data:
        if counter[label] < counter_goal[label]:
            img.save(f'./cifar10-train/{label}/{counter[label]:03d}.jpg')
            counter[label] += 1

        # exit loop early if reached goal number of images for ALL labels
        if counter == counter_goal:
            break
