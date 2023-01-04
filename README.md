# Arcface-for-Face-Recognition
ArcFace is an open source SOTA model for facial recognition. It has shown outstanding performance of 99.82% accuracy in LFW dataset. A detailed paper explanation is available in my article [here](https://medium.com/analytics-vidhya/arcface-facial-recognition-model-2eb77080aa80).

## Training
To train arcface on any custom dataset:
* firstly the collected face dataset must be arranged in a particular format. Use [LFW](http://vis-www.cs.umass.edu/lfw/) as reference.
* A text and csv file of name pairs would be generated using the script provided inside 'preparing_data' folder. 
* Place the dataset in the path 'training/data/'
* change path to data and run the code in the notebook 'run.ipynb'

In this repo, The arcface model was trained on a small custom south asian face dataset and it yielded an accuracy of 98%
