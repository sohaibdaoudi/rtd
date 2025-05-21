.. highlight:: none

.. _timeseriesproject_documentation:

🔋⚡ Intelligent Energy Management System with V2G Technology
================================================================================

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: ./LICENSE.txt
   :alt: License: MIT
.. image:: https://img.shields.io/badge/python-3.x-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version
.. image:: https://img.shields.io/badge/streamlit-1.20%2B-ff69b4.svg
   :target: https://streamlit.io/
   :alt: Streamlit Version
.. image:: https://img.shields.io/badge/MATLAB%2FSimulink-R20XXx-orange.svg
   :target: https://www.mathworks.com/products/matlab.html
   :alt: MATLAB/Simulink

🚧 Project Status: Under Active Development 🚧
-------------------------------------------------
**Attention:**
This application currently uses first-version models trained on limited data without standardization. An improved version will be released soon.

Notebooks Overview
--------------------

Our project includes several Jupyter notebooks that demonstrate various aspects of our data analysis and modeling approach:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Notebook
     - Description
   * - *Final_Project_Simulation.ipynb*
     - Main project simulation and integration of all components
   * - *Load-DeepLearning.ipynb*
     - Development of RNN models for load forecasting
   * - *Load-StatisticalStudy.ipynb*
     - Statistical analysis of load patterns and trends
   * - *SolarEnergy-DeepLearning.ipynb*
     - BiLSTM models for solar energy production forecasting
   * - *SolarEnergy-StatisticalStudy.ipynb*
     - Statistical analysis of solar generation data
   * - *Cars_Energy_dispo-DeepLearning.ipynb*
     - RNN models for V2G energy availability prediction

📋 Quick Links
---------------
- `Overview <#overview>`_
- `V2G Technology Explained <#v2g-explained>`_
- `Key Features <#key-features>`_
- `Notebooks Overview <#notebooks-overview>`_
- `Streamlit App Components <#streamlit-app-components>`_
- `Technology Stack <#technology-stack>`_
- `Simulation Details <#simulation-details>`_
- `Repository Structure <#repository-structure>`_
- `Installation & Usage <#running-the-application>`_
- `Acknowledgements <#acknowledgements>`_
- `License <#license-section>`_
- `Contact <#contact>`_

.. _overview:

Overview
--------

This project presents an intelligent energy management system that leverages Vehicle-to-Grid (V2G) technology to optimize electricity production costs. Using deep learning methods, we forecast key energy components and provide decision support for when to utilize stored energy from Electric Vehicles (EVs) versus supplementing with diesel generation.

The system integrates real-world solar irradiance data from Meknès, Morocco, along with residential load profiles representative of Moroccan consumption patterns. Our Streamlit application delivers weekly forecasts and cost-optimization recommendations based on comprehensive time series analysis.

.. _v2g-explained:

V2G Explained
-------------

What is Vehicle-to-Grid (V2G)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vehicle-to-Grid (V2G) is a cutting-edge technology enabling bidirectional energy flow between electric vehicle batteries and the power grid. Unlike conventional EV charging, V2G allows vehicles to not only consume electricity but also return it to the grid when beneficial.

.. image:: https://github.com/user-attachments/assets/f0aac5e9-e10b-41d8-8296-e3e35398dc79
   :alt: V2G Technology Diagram

.. _streamlit-app-components:

Streamlit App Components
------------------------

Our interactive web application provides a user-friendly interface for energy management decision support:

Key App Modules
~~~~~~~~~~~~~~~

- *data_utils.py*: Functions for loading and preprocessing time series data
- *model_utils.py*: Handles model loading and prediction generation
- *optimization.py*: Implements cost optimization algorithms
- *visualization.py*: Creates interactive charts and data visualizations
- *utils.py*: General utility functions used throughout the application

App Features
~~~~~~~~~~~~

- Weekly forecasts for load, solar production, and V2G availability
- Cost comparison between V2G utilization and diesel generation
- Interactive visualizations of energy patterns
- Optimization recommendations for energy sourcing

When connected to a V2G-enabled charging station, the system manages electricity flow through:

1. *Charging Mode:* The EV draws power from the grid to charge its battery
2. *Discharging (V2G) Mode:* The EV feeds stored energy back to the grid when:
   - Grid demand peaks
   - Energy prices are high
   - Renewable energy generation is low

This process is coordinated through sophisticated control systems with EV owner consent, typically including financial incentives for participation.

V2G Use Cases
~~~~~~~~~~~~~

V2G technology offers multiple benefits across the energy ecosystem:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Use Case
     - Description
   * - *Peak Shaving*
     - Reduces grid strain during high-demand periods by supplying stored EV energy
   * - *Grid Stabilization*
     - Provides ancillary services like frequency regulation and voltage support
   * - *Renewable Integration*
     - Stores surplus renewable energy for use when generation is insufficient
   * - *Emergency Backup*
     - Offers potential backup power during outages
   * - *Economic Optimization*
     - Creates revenue opportunities for EV owners through energy arbitrage

Our project specifically focuses on cost optimization by determining when V2G utilization is more economical than alternative generation sources.

.. _key-features:

Key Features
------------

- *Advanced Time Series Forecasting:* Deep learning models (RNN, BiLSTM, GRU) for accurate prediction of energy components
- *Localized Data Integration:*
  - Actual solar irradiation data from Meknès, Morocco
  - Realistic residential load profiles mirroring Moroccan consumption patterns
- *Cost Optimization Engine:* Algorithms to determine optimal energy sourcing strategies
- *Interactive Decision Support:* Streamlit application providing:
  - Weekly energy forecasts
  - Cost-comparison between V2G and diesel generation
  - Actionable recommendations for energy management
- *Comprehensive Analytics:* Data visualization and insights on load patterns, solar production, and V2G availability

.. _technology-stack:

Technology Stack
----------------

Deep Learning Models
~~~~~~~~~~~~~~~~~~~~
- TensorFlow/Keras
- Recurrent Neural Networks (RNN)
- Bidirectional LSTM (BiLSTM)
- Gated Recurrent Units (GRU)

Data Processing & Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~
- Python 3.x
- Pandas
- NumPy
- Scikit-learn

Web Application
~~~~~~~~~~~~~~~
- Streamlit

Data Sources
~~~~~~~~~~~~
- Solar irradiation measurements from Meknès, Morocco
- Synthetic residential load profiles based on Moroccan consumption patterns

.. _repository-structure:

Repository Structure
--------------------

.. code-block:: text

   Intelligent-Energy-Management-System-with-V2G-Technology/
   ├── Notebooks/                             # Jupyter notebooks for data analysis
   │   ├── Cars_Energy_dispo-DeepLearning.ipynb     # V2G energy availability modeling
   │   ├── Final_Project_Simulation.ipynb           # Complete project simulation
   │   ├── Load-DeepLearning.ipynb                  # Load forecasting with deep learning
   │   ├── Load-StatisticalStudy.ipynb              # Statistical analysis of load data
   │   ├── SolarEnergy-DeepLearning.ipynb           # Solar energy prediction models
   │   └── SolarEnergy-StatisticalStudy.ipynb       # Statistical analysis of solar data
   │
   ├── Best Models/                           # Optimized predictive models
   │   ├── best_model_BILSTM_SolarEnergy.h5       # BiLSTM model for solar prediction
   │   ├── best_model_RNN_LOAD.h5                 # RNN model for load forecasting
   │   └── RNN_CarsEnergy_v2g.h5                  # RNN model for V2G availability
   │
   ├── Datasets/                              # Raw and processed datasets
   │   ├── Solar_energy_cleaned.csv             # Processed solar energy data
   │   ├── Total_Load.csv                       # Load profile dataset
   │   └── total_power_EV_disponible.xlsx       # Available EV power data
   │
   ├── App_version_one/                       # Streamlit application
   │   ├── app.py                             # Main application entry point
   │   ├── data_utils.py                      # Data processing utilities
   │   ├── model_utils.py                     # Model loading and prediction functions
   │   ├── optimization.py                    # Cost optimization algorithms
   │   ├── utils.py                           # General utility functions
   │   ├── visualization.py                   # Data visualization components
   │   ├── style.css                          # Custom CSS styling
   │   ├── requirements.txt                   # App-specific dependencies
   │   │
   │   ├── data/                              # Application data files
   │   │   ├── Solar_Energy.xlsx
   │   │   ├── Total_Load.xlsx
   │   │   └── total_power_EV_disponible.xlsx
   │   │
   │   └── models/                            # Deployed model files
   │       ├── best_model_GRU_solar.h5
   │       ├── best_model_name_V2G_EV_energy_dispo.h5
   │       └── Load_Best_model_15.h5
   │
   ├── README.md                              # Project documentation (this file's source)
   └── LICENSE.txt                            # MIT License

.. _simulation-details:

Simulation Details
------------------

Our project leverages deep learning techniques to model and forecast three critical components of the V2G ecosystem:

Key Components
~~~~~~~~~~~~~~
- *Load Forecasting:* Predicting residential electricity demand using RNN models
- *Solar Energy Production:* Forecasting solar energy generation with BiLSTM/GRU models
- *V2G Availability:* Modeling available power from electric vehicle fleets

Data Inputs
~~~~~~~~~~~
- *Solar Irradiance:* High-resolution measurements from Meknès region
- *Residential Load:* Load profile data reflecting Moroccan consumption patterns
- *EV Fleet Parameters:* Available power data from electric vehicles

Forecasting Outputs
~~~~~~~~~~~~~~~~~~~
- Total Residential Load (kW)
- Solar Energy Production (kW)
- Available V2G Power Capacity (kW)
- Cost Optimization Metrics (MAD/kWh)

.. _running-the-application:

Running the Application
-----------------------

Prerequisites
~~~~~~~~~~~~~
- Python 3.7+
- Git

Installation Steps
~~~~~~~~~~~~~~~~~~
1. *Clone the repository:*

   .. code-block:: bash

      git clone https://github.com/yourusername/Intelligent-Energy-Management-System-with-V2G-Technology.git
      cd Intelligent-Energy-Management-System-with-V2G-Technology

2. *Set up a Python virtual environment:*

   .. code-block:: bash

      python3 -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. *Install Streamlit app dependencies:*

   .. code-block:: bash

      cd App_version_one
      pip install -r requirements.txt

4. *Launch the Streamlit application:*

   .. code-block:: bash

      streamlit run app.py

5. *Access the application:*
   Open your browser and navigate to http://localhost:8501

.. _acknowledgements:

Acknowledgements
----------------
This project draws inspiration from the "24-hour Simulation of a Vehicle-to-Grid (V2G) System" example provided by MathWorks. We extend our gratitude to MathWorks for providing this valuable conceptual foundation. The original simulation concept can be found `here <https://www.mathworks.com/help/sps/ug/24-hour-simulation-of-a-vehicle-to-grid-v2g-system.html>`_.

We also acknowledge the following resources that contributed to this project:
- TensorFlow/Keras documentation for deep learning implementation
- Streamlit documentation for web application development
- Various academic papers on V2G optimization techniques

.. _license-section:

License
-------
This project is licensed under the `MIT License <./LICENSE.txt>`_ - see the LICENSE.txt file for details.

.. _contact:

Contact
-------

Project Maintainers
~~~~~~~~~~~~~~~~~~~
- *Sohaib Daoudi*
  - Email: `soh.daoudi@gmail.com <mailto:soh.daoudi@gmail.com>`_
  - GitHub: `@sohaibdaoudi <https://github.com/sohaibdaoudi>`_

- *Marouane Majidi*
  - Email: `majidi.marouane0@gmail.com <mailto:majidi.marouane0@gmail.com>`_
  - GitHub: `@marouanemajidi <https://github.com/marouanemajidi>`_

----

.. raw:: html

   <p align="center">
     <em>Powering the future, one vehicle at a time.</em>
   </p>

.. This toctree directive is essential for navigation in Sphinx.
.. If you have other .rst files for different sections (e.g., api.rst, usage.rst),
.. list them here. Otherwise, for a single-page documentation, it can be minimal or omitted
.. if all content is on this page.
..
.. .. toctree::
..    :maxdepth: 2
..    :caption: Contents:
..
..    self