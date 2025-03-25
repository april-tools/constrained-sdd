from setuptools import setup, find_packages

setup(
    name="constrained-sdd",  # Replace with your package name
    version="0.1.0",  # Update version as needed
    author="Leander Kurscheidt",
    description="A Python package for working with constrained Stanford Drone Dataset.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/april-tools/constrained-sdd",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.2.4",
        "scikit-learn>=1.6.1",
        "pytest>=8.3.5",
        "requests>=2.32.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)