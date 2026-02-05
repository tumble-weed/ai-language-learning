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
Ds = 7

Au = 5

Show user sentences of Ds 5
if user gives answer to the sentence/exersice correctly{
increase au by 0.2(any constant)
}

if incorrect answer is given, decrease the Au

User 1: Mar->Tel: 


User 2: Hindi->Tel: 





Early Phase: (No user Data)

Middle Phase: ( Have some data, can predict whether user is able to find out the whether user can answer correctly or not.)
 
Late Phase: ( User imporved. set of sentence, which were difficult earlier, becomes easy now.)


TODO: 
Exploitation vs Exploration Trade Off


Day-Rating-Difficulty-Result
1-5-7-Hard
2-5.1-7-Hard
5-6-7-Medium
10-7-7-Above Easy
15-8-7-Easy


Rating of user, Correctness -> True Difficulty
5, 0 -> 6


Space repetition:

1. We have set of sentences (let's say 10) arranged according to their difficulty.
2. User gets sent1. User answers to the sentence correctly. Its elo increases a bit.
3. The sent1 is placed behind all the remaining sentences. ( This position is decided by the anki algorithm )
4. User gets sent2. User answers to the sentnce incorrectly. Its elo decreases a bit.
5. The sent2 is placed behind all the remaining sentencs accrodingly.
6. This process continues until all 10 sentences are answered atleast 2 times correctly.
7. Based on current elo, load new sentences.

"""

"""
pool contains 10 sentences.

user answers sent 1 correctly, hence sentence 1 will be available for display again after 60 seconds.

user answers sent 2 incorrectly, hence sentence 2 will be available for display again after 30 seconds.

-- 30 seconds later --

user gets sent 2 as 6th or 7th sentence again. answers it correctly this time. it will be available again after 60 seconds.

-- 30 more seconds later --

user gets sent sentence 1 again as 8th or 9th sentence. answers it correctly this time. it will be availabe again after 120 seconds.


Issues:
1. We run out of sentences.
Ans) Depends on dataset. If there are enough sentences, this should not be an issue.

2. User elo increased, how often to show easier sentences?
Ans) 

3. Sentence repetition timing management



TODO:
Add feature to automatically evalutate the user answers using llm and log the whole process using langfuse.



1. Reference answer is active voice, user answer is passive voice. We want to mark it incorrect since
user hasn't understood the concept of voice properly. But embedding systems will give high similarity score.


Give the user answer to llm for its evalutation. 
Prompt llm to tell whether the user answer matches with refernce answer or not.
if not, give proper reason. 

maybe use tools like spacy and fuzzy string match.

1. Adding word frequency.
2. cost analysis runpod vs openrouter.
if(runpod cost < openrouter cost){
    run embedding model locally for eval function and have less call to openrouter.
}
"""