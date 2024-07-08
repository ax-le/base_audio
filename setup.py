import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="base_audio",
    version="0.1.0",
    author="Marmoret Axel",
    author_email="axel.marmoret@imt-atlantique.fr",
    description="Package for handling audio files.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.inria.fr/amarmore/base_audio",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3.8"
    ],
    license='BSD',
    install_requires=[
        'librosa >= 0.6.0',
        'matplotlib >= 1.5',
        'numpy >= 1.8.0,<1.24',
        'IPython',
        'mir_eval',
    ]
)