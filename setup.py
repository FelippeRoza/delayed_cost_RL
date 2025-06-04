from setuptools import setup, find_packages

setup(
    name="safe_rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "matplotlib",
        "stable-baselines3",
        "scikit-learn",
        "joblib"
    ],
    entry_points={
        "console_scripts": [
            "train-safe-rl=scripts.train_safe_rl:main",
            "eval-safe-rl=scripts.eval_safe_rl:main"
        ]
    },
    python_requires=">=3.8",
)
