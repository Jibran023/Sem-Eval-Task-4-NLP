You can follow this readme file to work out your way for the submitted baseline files.

"Data_Provided" folder contains the data files provided by the task organizers and "Data_provided\Augmented_Synthetic_Data.jsonl" is the augmeneted data we produced.

Code contains two folders with code files for each task.

For Task A;
    - To run the Doc2Vec, run the doc2vec.py (You would have to modify the data files path accordingly)
    - To run the MLP model, run the mlp_model.py, (You would have to modify the data files path accordingly). You can test the model separately in the testing_model.py

For Task B;
    - Run the track_b2.py file to run the model, adjust the data paths accordingly.

"Code_provided" contains the code files provided by the task organizer which you can run to test the baselines for each task

These are the dependencies which need to be installed before running the programs:

'pip install sentence-transformers scikit-learn numpy scipy tqdm pandas matplotlib'
