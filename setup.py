from setuptools import setup, find_namespace_packages

setup(
    name='htr-pipeline',
    version='1.0.0',
    description='Handwritten text recognition pipeline, 02/27/2023 code rework.',
    author='Harald Scheidl',
    packages=find_namespace_packages(),
    url="https://github.com/githubharald/HTRPipeline",
    install_requires=['tensorflow==2.11.0',
                      'numpy==1.21.0',
                      'opencv-python',
                      'scikit-learn',
                      'editdistance',
                      'mathplotlib',
                      'path'],
    python_requires='>=3.8',
    package_data={'htr_pipeline.reader.stored_model': ['*']}
)
