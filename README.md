# Football Event Detection

This project aimed to develop a computer vision model capable of accurately detecting and classifying events such as plays, challenges, and throw-ins in football matches. 

We utilised the YoloX model for detecting the ball position in every frame, and quadratic interpolation was employed to identify missing frames. The SlowFast model was then used for classifying the events in each frame. 

The project’s primary objective was to automate event annotation to scale the data collection process and gather data from previously unexplored competitions. The final model achieved good accuracy in detecting and classifying events. 

The Kaggle competition and the data used can be found [here](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/overview).

# Results
![Football Events](https://github.com/felixboelter/football-event-detection/blob/main/data/event_output_gifs/538438_0_event.gif)


