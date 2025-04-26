from setuptools import setup, find_packages

setup(
    name="ridewise",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "joblib",
        "plotly",
        "requests",
        "python-dotenv",
        "pillow",
    ],
    author="RideWise Team",
    author_email="your.email@example.com",
    description="A machine learning application to predict ride-sharing surge pricing",
    keywords="ride-sharing, surge pricing, machine learning, prediction",
    url="https://github.com/yourusername/RideWise",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 