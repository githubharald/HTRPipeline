from setuptools import setup, find_namespace_packages

setup(
    name='htr-pipeline',
    version='1.2.0',
    description='Handwritten text recognition pipeline.',
    author='Harald Scheidl',
    packages=find_namespace_packages(include=['htr_pipeline', 'htr_pipeline.*']),
    url="https://github.com/githubharald/HTRPipeline",
    install_requires=['numpy',
                      'onnxruntime',
                      'opencv-python',
                      'scikit-learn',
                      'editdistance',
                      'path'],
    python_requires='>=3.8',
    package_data={'htr_pipeline.models': ['*']}
)
