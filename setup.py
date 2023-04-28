from io import open
from setuptools import find_packages, setup

setup(
    name="code_bert_score",
    version='0.3.5',
    author="Shuyan Zhou, Uri Alon, Sumit Agarwal, and Graham Neubig",
    author_email="urialon1@gmail.com",
    description="PyTorch implementation of Code BERT score",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='BERT NLP deep learning google metric',
    license='MIT',
    url="https://github.com/neulab/code-bert-score",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=['torch>=1.0.0',
                      'numpy',
                      'pandas>=1.0.1',
                      'requests',
                      'tqdm>=4.31.1',
                      'matplotlib',
                      'transformers>=3.0.0'
                      ],
    entry_points={
        'console_scripts': [
            "code-bert-score=code_bert_score_cli.score:main",
            "code-bert-score-show=code_bert_score_cli.visualize:main",
        ]
    },
    include_package_data=True,
    python_requires='>=3.6',
    tests_require=['pytest'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],

)
