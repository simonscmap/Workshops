# CBIOMES Annual e-Meeting 2020
The CMAP session during the CBIOMES annual meeting 2020 is centered around contrasting model estimates against in-situ observations. The idea is to identify and compile all measurements of prochlorococcus and synechococcus bacteria recorded in CMAP database and compare them with their corresponding counterparts estimated by the Darwin ecological model.<br/>

The CMAP session spans over a two-day period. During the first day, we will learn about the CMAP API and a selected list of methods required to identify and retrieve prochlorococcus and synechococcus cyanobacteria (see [introduction/CBIOMES2020.ipynb](introduction/CBIOMES2020.ipynb)). Using these methods we filter and retrieve measurements of prochlorococcus and synechococcus in form of a series of csv files (see [python/compile.py](python/compile.py)). <br/>

During the second day, the retrieved observations are clolocalized with the Darwin estimates ([python/simpleLocalizer.py](python/simpleLocalizer.py)) and we generate methods to quantify the consistency between the model estimates and in-situ observations ([python/comparePlot.py](python/comparePlot.py)).  <br/>

Finally, we will create a summary report for the plenary session on the last day of the meeting, [here](https://docs.google.com/presentation/d/1OWqzkvIyW4mf5UoDX7hS8rD8N_1kIt9JvtGBfts2F5U/edit#slide=id.p1).