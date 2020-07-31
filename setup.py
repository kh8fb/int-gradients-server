from setuptools import setup

setup(
    setup_requires=['setuptools_scm'],
    name='intgrads',
    entry_points={
        'console_scripts': [
            'intgrads=intgrads.serve:serve'
        ],
    },
    install_requires=[
        "click",
        "click_completion",
        "logbook",
        "flask",
        "torch",
        "transformers",
    ],
)
