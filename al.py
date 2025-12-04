"""
Algorithm for annotating sentences with difficulty levels.

1. Start

2. Get Sentences from uploaded csv

3. Annotate sentences with difficulty. (use UI for human based annotation)

4. Save annotated sentences to a file.

5. Stop

"""


"""

Algorithm for active learning based regression model training.

1. Start

2. Load annotated sentences from file.

3. Start training the regression model using annotated sentences.

4. While there are uncertain sentences:
    a. Get the uncertain sentences
    b. Annotate uncertain sentences
    c. Retrain the regression model

5. Stop

"""


"""
1. Solve page reurn issue
2. Review the audio loading logic
3. Modify UI of annotate tab


Later:
1. Download the audio and csv from cloud.
2. After annotation, upload the annotated csv to cloud.


gdown
rclone




"""