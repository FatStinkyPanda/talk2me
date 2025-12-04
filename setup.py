from setuptools import setup, find_packages

setup(
    name="talk2me",
    version="1.0.0",
    description="A fully offline, self-contained voice interaction system featuring speech-to-text, text-to-speech with voice cloning, and configurable wake word detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Talk2Me Team",
    author_email="info@talk2me.example.com",
    url="https://github.com/talk2me/talk2me",
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
            "talk2me=src.main:main",
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