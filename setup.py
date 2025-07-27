import setuptools

# Read the contents of your requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="doculuma",
    version="0.1.0",
    author="Vansh Patel", # You can change this
    author_email="vanshfpatel@gmail.com", # You can change this
    description="A tool for document ingestion, versioning, and analysis.",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires=">=3.11",
)