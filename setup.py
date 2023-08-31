from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    description = f.read()


setup(
    name='NeuralNetworksTrainingPackage',
    version='1.0.0',
    description='Set of scripts for automated training of models from non-homogenous-paper',
    long_description=description,
    url='https://github.com/Bartosz-G/NeuralNetworksTrainingPackage.git',
    author='BartoszGawin',
    author_email='gawinbartosz@icloud.com',
    packages=find_packages(),
    zip_safe=False,
    install_requires=['numpy', 'pandas', 'scikit-learn', 'torch', 'torcheval']
)
