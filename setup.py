from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="talk2me",
    version="1.0.0",
    description="A fully offline, self-contained voice interaction system featuring speech-to-text, text-to-speech with voice cloning, and configurable wake word detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daniel A Bissey",
    author_email="support@fatstinkypanda.com",
    url="https://github.com/FatStinkyPanda/talk2me",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "vosk>=0.3.45",
        "TTS>=0.22.0",
        "pyyaml>=6.0",
        "numpy>=1.21.0",
    ],
    entry_points={
        "console_scripts": [
            "talk2me=talk2me.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
)
