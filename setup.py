from setuptools import setup
from setuptools import find_packages


def readme():
    with open('README.md', encoding='utf-8') as file:
        return file.read()

def license():
    with open('LICENSE.txt', encoding='utf-8') as file:
        return file.read()

setup(
    name='nalaf',
    version='0.2.0',
    description='Pipeline for NER of natural language mutation mentions',
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Topic :: Text Processing :: Linguistic'
    ],
    keywords='nlp ner natural langauge crf svm extraction entities relationships framework',
    url='https://github.com/Rostlab/nalaf',
    author='Aleksandar Bojchevski, Carsten Uhlig, Juan Miguel Cejuela',
    author_email='i@juanmi.rocks',
    license=license(),
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'nltk',
        'beautifulsoup4',
        'requests>=2.8.1',
        'python-crfsuite>=0.8.4',
        # Note: it may cause problems on Windows machines
        # Throubleshooting
        # * Install python3-devel package or similar from your UNIX distribution

        # 'gensim'
        # Note: it may cause problems on different environments (e.g. Travis CI)-- install it manually

        # 'spacy'
        # Note: it may cause problems on different environments (e.g. Travis CI)-- install it manually
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
    setup_requires=['nose>=1.0'],
)
