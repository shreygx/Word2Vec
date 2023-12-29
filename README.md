galytix/
|-- your_module/
|   |-- __init__.py
|   |-- word_embeddings.py
|   |-- phrase_similarity.py
|-- tests/
|   |-- __init__.py
|   |-- test_word_embeddings.py
|   |-- test_phrase_similarity.py
|-- setup.py
|-- requirements.txt
|-- main_script.py

I am uploading the link of word2vec model which can be dowloaded and directly can be run in the galytix.py file with the phrases.csv file.
model:https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM
The file galytix.py is the fully compiled python file which can be executed on google colab or any other IDE.  
I am also uploading the galytix.ipynb file in which you can see the executed results.

I have used the data in batches for less memory usage and faster processing.

We can further optimize the code in future by modularization , optimization:Explore batch processing, parallelization, and efficient NumPy usage for computational efficiency.
We can make the code more efficient by removing the stop words and enhancing the model even further.


