#!/usr/bin/env python3

import setuptools

install_requires = [
        ]

setuptools.setup(
    name="discord_transformer",
    python_requires=">=3.8.10",
    description="Explorative ML with GPT-2 over discord chat data",
    version="0.1",
    author="Kolja Hopfmann",
    author_email="k.hopfmann@hotmail.de",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    )