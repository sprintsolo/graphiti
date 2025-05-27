from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="graphiti-core",
    version="0.11.6",
    author="Paul Paliychuk, Preston Rasmussen, Daniel Chalef",
    author_email="paul@getzep.com",
    description="A temporal graph building library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sprintsolo/graphiti",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10,<4",
    install_requires=[
        "pydantic>=2.8.2",
        "neo4j>=5.23.0",
        "diskcache>=5.6.3",
        "openai>=1.53.0",
        "tenacity>=9.0.0",
        "numpy>=1.0.0",
        "python-dotenv>=1.0.1",
        "pyairtable>=3.1.1,<4.0.0",
    ],
    extras_require={
        "anthropic": ["anthropic>=0.49.0"],
        "groq": ["groq>=0.2.0"],
        "google-genai": ["google-genai>=1.8.0"],
        "dev": [
            "mypy>=1.11.1",
            "pytest>=8.3.3",
            "pytest-asyncio>=0.24.0",
            "pytest-xdist>=3.6.1",
            "ruff>=0.7.1",
        ],
    },
) 