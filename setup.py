from setuptools import setup, find_packages

setup(
    name="speaker_id_gender_classification",
    version="1.0.0",
    description="A research project for Speaker Identification and Gender Classification",
    author="Mostafa Kermani Nia",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "librosa",
        "gdown",
        "xgboost",
        "tensorflow",
        "noisereduce",
        "soundfile",
    ],
)
