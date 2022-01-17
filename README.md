

<br />

  <h1 align="center">NLP Exam 2021 </h1> 
  <h2 align="center">MSc Cognitive Science, Aarhus University </h2> 
  <p align="center">
    Frida Hæstrup & Jan Kostkan
    <br>
</p>

<br>   
<br>   


This repository contains code related to our exam project in the Autumn 2021 course in Natural Language Processing, MSc Cognitive Science, Aarhus University. 

### Project structure

The repository has the following directory structure:

```bash
emodynamics-dungeons-and-dragons/  
├── reproduce_analysis.sh #running analysis
├── checks_windows.sh #running check on window size
├── checks_bins.dh #running check on bin size
├── emissions.csv #estimated carbon emissions from analysis
├── src/  #folder with analysis scripts 
│   └── analysis/ #models
│   │   └── ...
│   └── entropies/ #infodynamics
│   │   └── ...
│   └── binning.py #timebinning
│   └── classification.py #emotion classification
│   └── ntr.py #calculate novelty, transience, resonance
│   └── util.py #utility functions
```


### Environmental impact
The estimated emissions of this analysis can be seen in [emissions.csv](emissions.csv). Power consumption of this analysis and its experiments was 0.017 kWh. Based on [Codecarbon](https://github.com/mlco2/codecarbon)’s estimation of carbon efficiency in Denmark, where the analysis was run, the total emissions are estimated to be 0.01 kgCO_2 eq.
