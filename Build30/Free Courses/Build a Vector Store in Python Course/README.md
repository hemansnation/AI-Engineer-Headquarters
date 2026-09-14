# Build a Vector Store from Scratch in Python

A free text course by [Himanshu Ramchandani](https://www.linkedin.com/in/hemansnation/) (@hemansnation).

Today we are building a vector store. We are using Python to build everything. It is easy to make and easy to understand, because we are not using a lot of different libraries. We are using very few things, so you can see exactly what is going on.

You probably heard of a vector store before. By the end of this, you will have built one.

This is the written version of the live session, and it follows the same order. Read it on its own, or read it alongside the video.

---

## What you need

Python, an editor, and a terminal. I use VS Code. Use any editor you like.

One package gets installed, and it is free and open source. No API keys, no paid model, no account anywhere.

Slide references point at the workshop deck. If you have the PDF open, the numbers line up.

**Code:** `workshops-and-webinars/september/build-vector-store-in-python/` in the [AI Engineer Headquarters](https://github.com/hemansnation/AI-Engineer-Headquarters) repo.

[The PPT for Diagrams and Architectures here](https://docs.google.com/presentation/d/1DZ0ZjTHAodN-YXOehqvHHekzyg4EzJ7jawlaXIadCtk/edit?usp=sharing)

---

## Table of contents

**Concepts**

- [1. What we are building](#1-what-we-are-building)
- [2. Keyword search vs semantic search](#2-keyword-search-vs-semantic-search)
- [3. What a vector is](#3-what-a-vector-is)
- [4. Why we convert English into numbers](#4-why-we-convert-english-into-numbers)
- [5. Tokens](#5-tokens)
- [6. The embedding model](#6-the-embedding-model)
- [7. How close are two sentences](#7-how-close-are-two-sentences)

**Setup**

- [8. Setting up the project](#8-setting-up-the-project)
- [9. Creating and activating the environment](#9-creating-and-activating-the-environment)
- [10. Hugging Face and the model page](#10-hugging-face-and-the-model-page)
- [11. The old way with Word2Vec and GloVe](#11-the-old-way-with-word2vec-and-glove)
- [12. Installing sentence-transformers](#12-installing-sentence-transformers)

**Embeddings**

- [13. Writing the dataset](#13-writing-the-dataset)
- [14. Turning sentences into vectors](#14-turning-sentences-into-vectors)
- [15. Running it for the first time](#15-running-it-for-the-first-time)
- [16. NumPy array vs Python list](#16-numpy-array-vs-python-list)

**The math**

- [17. Dot product](#17-dot-product)
- [18. Magnitude](#18-magnitude)
- [19. Cosine similarity](#19-cosine-similarity)
- [20. Testing the function](#20-testing-the-function)

**The store**

- [21. The VectorStore class](#21-the-vectorstore-class)
- [22. The add method](#22-the-add-method)
- [23. The query method](#23-the-query-method)

**Running it**

- [24. Putting it all together](#24-putting-it-all-together)
- [25. Running your first query](#25-running-your-first-query)
- [26. Changing the query](#26-changing-the-query)

**After that**

- [27. Making it survive a restart](#27-making-it-survive-a-restart)
- [28. Where this goes next](#28-where-this-goes-next)
- [29. One question from the session](#29-one-question-from-the-session)
- [The full file](#the-full-file)

---

## 1. What we are building

> Deck: slides 01 to 03

Before we start building, we need to understand some of the parts. Those parts help us understand how the thing works, and then we can build it in Python.

Here is exactly what we are building. It might look confusing as a first timer, but this is the whole flow.

There will be some text content. We use an embedding model to convert that into embeddings, or vector representation. We put them in our vector store. Then we use a query. It can be any query, whatever we pass. It goes through the embedding model, it searches, it runs the cosine similarity loop, and it gives me the answers.

Simple. Two phases. One phase runs once when you load your data. The other runs every time somebody searches.

[Back to top](#table-of-contents)

---

## 2. Keyword search vs semantic search

> Deck: slide 04

When you talk about keyword based search or semantic search, we need to find the difference.

Think about Google before LLMs, when there was no semantic search. You searched something and Google gave you a response based on the keywords you typed. That is the first thing they did. You search "laptop", and if any website contains the word "laptop" exactly, it lists that website in front of you.

That is not how Google works today. Today they work on semantic search. Semantic search means Google gives you a relevant result based on the context and the intent of your query. It is not only keyword. It is also intent and context.

**Here is the example we use all the way through.**

My query: `A kitten is playing with a toy.`

What I have stored: `The cat sat on the mat.`

Now run it through a keyword based search. Compare the words. Not one word in my query matches anything in the store.

Some of those words never carry weight anyway. Words like "is", "a", and "with" are stop words. Stop words do not help in generating the context of a query. That comes from our grammar, and it makes no sense to a machine. So the search keeps the important words, which are kitten, playing, and toy.

Then it checks the tokens on the other side, which are cat, sat, and mat. Those are not equal in keyword based search. There is no shared word, so there is nothing to rank. If Google worked on keywords only, you would find nothing here.

**Now the semantic side.**

You have a query, and the query gets converted into a vector representation, which is a numerical representation. Your store also holds a stored vector. Run cosine similarity between the two and you come up with a score, and that score tells you how close they are.

If you plot those two vectors in 2D space on a graph, the coordinates land closer to each other.

As a human, you understand that the query and the stored value are similar. I am talking about a kitten, and a cat is also called a kitten. Playing with a toy and sat on the mat are both close. Kitten and cat sit close to each other, so cosine similarity in that case comes out close.

This is why LLMs work really well when you ask something without the exact keywords. I am searching for a song and I do not know the exact lyrics. Even if I am somewhat close, it still gives me a good result and finds the song. That happens because of the vector representation of my query and of whatever sits stored in the database. That cannot happen in keyword based search.

[Back to top](#table-of-contents)

---

## 3. What a vector is

> Deck: slide 05

A vector is still a list of numbers.

If I am talking about a list of numbers, I can store those numbers like this. If you know about Python lists, you can store them from the zeroth index onward.

```python
vector = [0.2, -0.5, 0.9]
```

Now think about plotting these numbers on a graph.

Stock market, kitten, and cat. Stock market sits far away from the other two, because a stock market is not related to a cat. Cat and kitten sit close. Mat and sofa sit close.

These vector representations carry similarity based on how close those words sit in English. If I treat the numbers as probability, cat and kitten are close to each other. Mat and sofa are close to each other. Stock market is not close to either.

That is how you represent any English word as a numerical representation in a 2D plot. You can do that yourself with some Python plotting functions.

[Back to top](#table-of-contents)

---

## 4. Why we convert English into numbers

> Deck: slide 06

If you do not convert an English word into a vector, you do not get semantic search, and you do not get any semantic meaning out of the text.

Machines do not understand English. They only understand numerical values. That is why we convert everything into numbers. Even when you press a key on the keyboard, it gets converted into an ASCII value, and then into bits, into binary zeros and ones.

English makes no sense to a computer. That is why we need mathematics, and that is why we need to convert everything into numerical representation.

There is a concept called embedding. If I need to convert a particular word into its vector representation, that is what embedding means. We convert it into a numerical representation.

Take ASCII codes. Capital A has an ASCII code of 65. You can call that a standard way of converting an English character into a number.

In a similar way, I need to do that with text. "The cat sat on the mat", "a dog is playing in the park", "stock market crashed", all of those. There are a lot of different algorithms and models available to convert them into vector representation, depending on how complex they are.

[Back to top](#table-of-contents)

---

## 5. Tokens

> Deck: slide 06

You already know about the OpenAI tokenizer. Different tokenizers use different algorithms, and you can use whichever one fits what you want.

Take "The cat sat on the mat". There are seven tokens in it. Tokens are smaller parts of your sentence.

Now think about it. You could give a number to each English word in a dictionary. There are 250,000 total words in English. But those numbers make no sense on their own, because assigning one number to each word means you cannot pull context from it. The meaning is not there.

If you give 1 to apple, 2 to alphabet, and 3 to something else, you cannot extract context from that. So instead we convert text into tokens, because we need to extract context and intent, and there are different algorithms to do it.

Some algorithms use individual words. Some algorithms divide the whole word into sub words, depending on how complex the algorithm is. The word "neural" can split into "neu" and "ral", and each part gets a different vector representation. "Python" can split into "py" and "thon" the same way.

More advanced algorithms do not focus on individual words only, they focus on sub words as well. The better models you find all go for sub words.

[Back to top](#table-of-contents)

---

## 6. The embedding model

> Deck: slide 06

One of the embedding models that is freely available is **all-MiniLM-L6-v2**. You will see it commonly used in most applications.

Whatever query we use, and whatever database we use, we pass it through this model and it gives us a vector representation of the text.

For example, encode "The cat sat on the mat" and check the length of what comes back. The length is 384. Each position holds its own value, and those values are floats, decimal values.

Every model works differently. This one sits on Hugging Face, so we can use it directly. It converts your English words into vector representation.

You can install it very easily and use these functions, so everything gets done for you. We will do that in VS Code and code it out.

**Why not OpenAI?** Because those are not free. We want a free embedding model, and one of the free embedding models is all-MiniLM-L6-v2.

[Back to top](#table-of-contents)

---

## 7. How close are two sentences

> Deck: slides 07 and 15

Keep this in mind. Similar words sit close to each other.

Cat and mat sit close. Kitten and sofa sit close. Cat and mat against stock market are not close. Kitten and sofa against stock market are not close either.

If you find the difference between the vector representations in 2D space, that distance tells you how close they are. That is how you come up with cosine similarity, and cosine similarity does that for you.

If two vectors point opposite, the score goes one way. If they are related, they sit close to each other.

One thing to keep in mind. The number scale and the graph are for us. They are not for the machine. The machine does its work through numbers and calculations. We plot things out so that we can see what is happening, and so that we can confirm the machine got it right.

[Back to top](#table-of-contents)

---

## 8. Setting up the project

> Deck: slide 08

I already created a folder in my system, and that folder sits on GitHub. In the repo you will find `workshops-and-webinars`, then `september`, then the folder for today's code.

This is VS Code, and inside it I created a folder called `build-vector-store-in-python`. VS Code is an editor. You can use any editor here.

The coding setup requires very few things. Python, which you already know, and a terminal. In VS Code, go to Terminal and click New Terminal.

First, check the Python version on your system.

```bash
python --version
```

I get Python 3.14. Your version will be different, and you can definitely work with that.

[Back to top](#table-of-contents)

---

## 9. Creating and activating the environment

> Deck: slide 09

The first thing we do is create an environment. An environment means creating a folder for our project.

**Why it matters.** If project A is there, project B is there, and a lot of other projects are there, and I do not create an environment, everything gets stored globally in my system.

Think about it. You are working with OpenAI APIs and you are using Gemini also. You are using LangChain, and you are using PyTorch in the background. Different projects have different libraries and frameworks, and we do not want to merge them. Different versions of different libraries are not compatible with each other, so they affect each other while running a project.

Think of it as a shared pool. Without a virtual environment you are running multiple projects inside that pool, so they do not work properly and they affect each other.

Instead, for each project we create a virtual environment, which is a folder. In that folder sit all the libraries, packages, and frameworks required to run that project. It can be LangChain, it can be PyTorch, it can be TensorFlow, whatever you are working on. If I create another project, I create a separate environment for that one.

**Create it.**

```bash
python3 -m venv venv
```

The first `venv` is the keyword. The second one is the name of the folder, and you can name it anything. September, December, apple, whatever you want. There should be no confusion, so we generally name it the same as the keyword.

Run that and a folder appears automatically in your project. Open it and you will find some configuration files, Python already installed in it, and pip already installed. We do not need to know what sits inside. It is just an environment, and it is not our work.

**Activate it.**

```bash
source venv/bin/activate
```

On Windows the command is `venv\Scripts\activate`.

Activation means that whatever I install now gets installed inside this folder. If I install TensorFlow, TensorFlow lands in this folder and nowhere else in my system. When I run something, this project folder gets used.

That is why we activate the environment. On your system there will be multiple environments, and I want this one only, so inside this folder I use my own environment.

[Back to top](#table-of-contents)

---

## 10. Hugging Face and the model page

> Deck: slide 10

Search for all-MiniLM-L6-v2 and you land on Hugging Face.

Hugging Face is similar to GitHub, but for models. LLMs, small language models, and a lot of different models. People use datasets, train their models, and upload them here, the same way we push code to GitHub. Anybody can use it, and everything there is open source.

Sentence Transformers sits there. They created a model that helps you create the embeddings for a text, and the name of the model is all-MiniLM-L6-v2.

It is a sentence transformer model. The name suggests what it does, which is transform a sentence into numerical representation. The card says it maps sentences and paragraphs to a 384 dimensional dense vector space. Vector space means your vector representation. It also says it can be used for tasks like clustering and semantic search, which is exactly what we are talking about.

So if I convert my text into vector representation using this model, I can run semantic search on my database.

On their main page you will find a lot of other models as well, not only sentence transformer models. Millions of downloads sit next to all-MiniLM-L6-v2, so people are using it. You will also find datasets there, free to use for your own work.

**One aside.** If you look at a BPE tokenizer instead, that is byte pair encoding, and different algorithms use different kinds of it. You can read about byte pair tokenization on Hugging Face. How it works is a different topic. Today we focus on all-MiniLM-L6-v2.

[Back to top](#table-of-contents)

---

## 11. The old way with Word2Vec and GloVe

> Deck: slide 06

Today we have LLMs and sentence transformers. Before that we worked on natural language processing.

There is a Word2Vec dataset available, and there is GloVe from Stanford. As I said, each word carries a fixed vector representation. You can go and download the model, and a text file comes with it.

I wrote a simple Python function around it. The function opens that file, finds the word I want, and gives me the vector embedding of that word.

Pass `python` and the output is an array of numbers, and those numbers are the vector representation of python. Pass `neural` and it gives you the vector representation of neural. Pass apple, pass laptop, pass any word, and it gives you the vector representation of that word.

Where do those words sit? In that dataset. Open the GloVe link and you will find everything inside it, global vectors for word representation. You can download it, look at how each word looks in vector space, and run the code yourself.

That is how vector representation works. The difference today is that more advanced algorithms do not focus on individual words only, they go for sub words as well. In this dataset, individual words carry the vector representation.

[Back to top](#table-of-contents)

---

## 12. Installing sentence-transformers

> Deck: slide 10

We use sentence transformers so that we can convert our own English sentences into vector representation and store them.

```bash
pip install sentence-transformers
```

This installs sentence transformers in my environment, not across my whole system. It is free, you can download it, and it is open source.

Once it finishes, check the libraries folder inside `venv` and you will see a lot of different files now. NumPy is there. Scikit-learn is there. Tokenizer, Torch, and Transformers, all of it.

We did not install those. So how are they there? Because whenever you install sentence transformers, it internally carries a lot of dependencies, and when you install it, all those dependencies get installed too. NumPy, joblib, and a lot of other libraries.

We do not want to go through those files. They are part of the environment, not part of our work.

For this project I need only one file, and I am calling it `main.py`.

```
build-vector-store-in-python/
    venv/
    main.py
```

[Back to top](#table-of-contents)

---

## 13. Writing the dataset

> Deck: slide 19

I do not want to make it complex, so I write my own dataset.

```python
sentences = [
    "The cat sat on the mat.",
    "The dog is playing in the park.",
    "The stock market crashed today.",
    "Investors are worried about inflation.",
    "The kitten is sleeping on the sofa.",
]
```

I am using fixed sentences because I built my code around them.

Now look at what I picked. Sentence one and sentence five sit close to each other. If I convert them into vector representation, the machine easily treats both of them as close. If I do not, it does not.

Similar to that, the third one and the fourth one sit close to each other, because we are talking about finance there. Crashed and inflation belong to the same world.

[Back to top](#table-of-contents)

---

## 14. Turning sentences into vectors

> Deck: slide 06

We already installed sentence transformers, so now I use it.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
```

`SentenceTransformer` is a class. Using this class you create an object, which is what is happening here. We can pass any kind of model, because this class carries multiple models, not only the single one we looked at.

Your editor will suggest things automatically as you type. Mine suggests through Copilot. I do not always want what it suggests, so read before you accept.

Now pass the sentences into the model.

```python
embeddings = model.encode(sentences)
```

This model has an `encode` function, and we can use it. We pass the sentences and it gives me the vector representation, a list of numbers. This `encode` function can turn any sentence into a vector.

Once that is done, `embeddings` holds everything for me, so let us print it and see how it looks.

```python
for sentence, vector in zip(sentences, embeddings):
    print(sentence)
    print("Vector length:", len(vector))
    print("First 5 numbers:", vector[:5])
    print("---")
```

`sentence` and `vector` are two variables I am using. There is no fixed variable name, these are the ones I picked. In `zip` I am passing `sentences` and `embeddings`, so both are available. Sentences are the text, embeddings are the vectors, the numerical values.

First we print the sentence. Then we print the length of the vector. Then we print the first five numbers of the vector to see how they look. Then some dashes.

The loop lines up the sentences and the embeddings and gives me the actual sentence one by one with the corresponding vector representation, plus the length, so I can see how big those vector representations are.

[Back to top](#table-of-contents)

---

## 15. Running it for the first time

> Deck: slide 20

```bash
python main.py
```

You may see a message about a Hugging Face token. That happens because the terminal needs to communicate with the Hugging Face servers to download the model.

Now look at the output. "The cat sat on the mat" comes back with its first five numbers, and the length is 384.

The length is fixed. The length of the vector stays the same for every sentence. The first five numbers are different in each sentence.

Now think about it. Individual words carry a big representation, depending on which algorithm you use. For a whole sentence, those numbers get really big. The length of the vector is 384, which means 384 numbers cover the whole sentence.

Different sentences get different results and different numbers. Similar sentences get similar numbers, or close numbers.

[Back to top](#table-of-contents)

---

## 16. NumPy array vs Python list

> Deck: slide 16

One more thing to keep in mind. The `encode` function gives you a NumPy array. You can convert that into a list.

```python
vector_as_list = list(vector)
```

This conversion matters because we need to use them as lists. We are going to perform list operations, so the vector gets converted into a list.

Arrays and lists are different. Array values sit contiguously in order, one after another, and they use neighbouring memory addresses to store the values. A list does not work like that. A list uses different positions in memory space, and it holds the memory addresses internally, so the values can sit at different positions.

That is the common difference between lists and arrays.

[Back to top](#table-of-contents)

---

## 17. Dot product

> Deck: slide 12

Now we need to know how the machine figures out that these sentences are similar to each other. For that we implement cosine similarity.

Cosine similarity uses mathematics, and it uses the dot product.

Vector A carries some values, vector B carries some values, and the product looks like this. You multiply those values and get the value.

I need to import math, because I need mathematics for this, and I need to implement dot product, magnitude, and cosine similarity individually.

```python
import math


def dot_product(vector_a, vector_b):
    total = 0.0
    for a, b in zip(vector_a, vector_b):
        total += a * b
    return total
```

I start at `0.0` because I am going to add all of those values in one place. The loop picks one value from vector A, one value from vector B, zips them, multiplies them, and adds the result into the total. That gives me the dot product of the whole thing.

[Back to top](#table-of-contents)

---

## 18. Magnitude

> Deck: slide 13

If I talk about the magnitude of the vector, the vector sits there, the square of each value sits there, you add those up, and the square root gives you the magnitude.

```python
def magnitude(vector):
    total = 0.0
    for value in vector:
        total += value ** 2
    return math.sqrt(total)
```

Those are exponents. You can write `value * value` instead, but `value ** 2` means value squared.

The last thing we do is use `math.sqrt` and pass the total inside it, so it takes the square root. That gives you a magnitude.

Take the vector `[3, 4]`. Square each and you get 9 and 16. Add them and you get 25. The square root is 5, and that 5 is the length of the arrow from the origin to the point [3, 4].

[Back to top](#table-of-contents)

---

## 19. Cosine similarity

> Deck: slides 14 and 15

Through these mathematical operations we know how a particular vector performs. They give you a number, so that you understand, and so that the machine understands, how close two things are.

**Keep these three values in mind.**

`1.0` means they point in the same direction.

`0.0` means they are unrelated, or at a right angle mathematically.

`-1.0` means they point in opposite directions. In practice, negative values generally do not happen with sentence transformers, but those are the values you can find whenever you perform this math.

```python
def cosine_similarity(vector_a, vector_b):
    dot = dot_product(vector_a, vector_b)
    mag_a = magnitude(vector_a)
    mag_b = magnitude(vector_b)

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)
```

The `dot_product` function gets called here. I need the magnitude of A and the magnitude of B. If the magnitude of A is zero or the magnitude of B is zero, I return `0.0`, and `0.0` means they are unrelated. That is how we know everything is fine and they are simply not related.

If they are related, we return the dot product divided by the magnitude of A times the magnitude of B.

The full formula looks like this. Dot product on top, the two lengths underneath, one divide, and you have a score.

We are not using third-party packages, but `math` is a Python module that ships with the language, so we can use it.

[Back to top](#table-of-contents)

---

## 20. Testing the function

> Deck: slide 15

Let us run one quick test. Put this at the end of the file.

```python
a = [1, 0]
b = [1, 0]
c = [0, 1]

print(cosine_similarity(a, b))
print(cosine_similarity(a, c))
```

A and B are similar, so that gives me 1, and they point in the same direction. A and C are not similar, so they sit at a right angle, and the output is zero.

Run it and you get 1 for the first and 0 for the second, which means it is working.

Now put the test back into comments and carry on.

[Back to top](#table-of-contents)

---

## 21. The VectorStore class

> Deck: slide 16

We need to create a class that uses both of those methods. It stores the text along with the vector. It takes a query when somebody asks something, and it tells us which database values sit closest to the user query. We can pass the query ourselves.

```python
class VectorStore:
    def __init__(self):
        self.items = []
```

For those who do not know, `__init__` is the constructor of a class. It assigns memory to every part of the class, and it gets called automatically whenever you use this class.

A class is an updated version of structures. If you want to store some kind of data in programming, you use `struct` in C, classes in C++, and classes in Python. You create your structures in your own way through classes.

Inside `__init__` you can create any kind of variables you want. Here I create `items`, and it is a list, nothing else. All the items get stored inside it in the form of dictionaries.

[Back to top](#table-of-contents)

---

## 22. The add method

> Deck: slide 16

```python
    def add(self, text, vector):
        self.items.append({
            "text": text,
            "vector": list(vector),
        })
```

`self` is always the first argument in a method of a class. Whenever you create a function inside a class you call them methods, and a method is a property of the class. You are adding an abstraction layer on top, so to use this you need to access it through an object.

You have already seen this. Go back and look at how we imported `SentenceTransformer`. How do you know it is a class? You always find capital letters as the initials. When you implement it, you create an object, and you use that object, `model`, to call a particular function. `encode` is a method inside that class.

Similarly, I created the `VectorStore` class and my own method inside it. To access it, I create an object from the class and then call `add` on that object.

`append` is part of lists in Python, so if you want to add something to a list, you can do that. `items` is a list, and I am using a dictionary for the entry. Text goes in as text, and vector goes in as vector.

This time we know the vector arrives as an array by default, so I convert it into a list first and store it in a Python list. Keep that in mind.

[Back to top](#table-of-contents)

---

## 23. The query method

> Deck: slides 17 and 18

I want a query function as well. When I query the database, it gives me the relevant result.

```python
    def query(self, query_vector, top_k=2):
        scored_items = []

        for item in self.items:
            score = cosine_similarity(query_vector, item["vector"])
            scored_items.append((score, item["text"]))

        scored_items.sort(key=lambda pair: pair[0], reverse=True)
        return scored_items[:top_k]
```

I set `top_k=2` because the dataset is small. `top_k` means that whenever it searches the vector store, it gives me the top two results. If the value is 5, it gives me the top five results. If the value is 15, it gives me the top fifteen.

You will see this in RAG as well. You divide your data into smaller parts, we call them chunks, and when you pass `top_k=5` it gives you the top five chunks whenever somebody queries it.

`scored_items` is a list. The loop walks `self.items`, calls cosine similarity with the query and the item vector, and gets a score. Once it gives me a score, I append the score first and then the text.

That appends a tuple. A tuple is another data structure, like a list, and it pairs these two values. The score corresponds to whatever the text is.

Then I sort. `lambda` is a one line function in Python. The argument comes in, and that particular part gets returned. Whatever value comes from my stored items, based on their keys, it gives me the zeroth value of that pair, and `reverse=True` sorts in descending order.

The last line returns items from the zeroth value onward. We are slicing the list, and the value of `top_k` is 2, so from 0 to 2, which means positions 0 and 1. The first two values get returned.

[Back to top](#table-of-contents)

---

## 24. Putting it all together

> Deck: slide 19

Here is a simple summary of what we did.

We wrote the dot product and the magnitude, because both are required to calculate cosine similarity. Cosine similarity tells me whether these vector representations point in the same direction or in the opposite direction. Then I created a vector store that adds the values after converting them into numerical representation. When somebody queries it, it gives me the top two relevant results based on cosine similarity. We already created the score, and based on that score it sorts, and whichever holds the highest score comes back in the top two.

Now I create one last thing.

```python
store = VectorStore()
```

I create an object of the `VectorStore` class. Now I can use `store` and call any function inside it. I created `query` and `add`, so I can call those and execute them.

```python
for sentence in sentences:
    vector = model.encode(sentence)
    store.add(sentence, vector)
```

`sentences` is what we created at the beginning. That is my database, so I need to convert it into vector representation first. Inside the loop I use `model.encode` and pass the sentence, so one by one it passes everything through. Then I use `store.add` to append everything, so the sentence along with its vector representation goes into the store.

The sentence comes in, the vector representation comes through the encode step, and the text and the vector go in together. Because we have a loop, it goes around those five sentences and their corresponding vector representations.

```python
query_text = "A kitten is playing with a toy."
query_vector = model.encode(query_text)
```

I need to encode the query as well, so that both of them sit in vector representation and the search runs easily in the vector database.

This always happens when you pass a query into ChatGPT. It gets converted into a vector embedding, because we need to search the vector store, not run a keyword based search over text.

```python
results = store.query(query_vector, top_k=2)
```

We already kept `top_k` at 2 by default, and we can pass 3, 4, 5, whatever we want. So `store` calls `query`, and `query` takes the query vector, jumps around the cosine similarity for each item, calculates it, and gives me the top two values inside `results`.

```python
print(f"Query: {query_text}")
print("Top 2 results:")
for score, text in results:
    print(f"  {score:.4f}  -  {text}")
```

I use `score` as a floating point number, and `text` as the sentence corresponding to it.

[Back to top](#table-of-contents)

---

## 25. Running your first query

> Deck: slide 20

Think about it before you run it. With this query, the cat part, which is the first sentence, and the last sentence should both sit close. Those two are the top two priorities, so those two sentences should come back.

```bash
python main.py
```

My query is "A kitten is playing with a toy." and the top two results are "The kitten is sleeping on the sofa" and "The cat sat on the mat", because their scores sit very close to each other.

We created our own vector store.

[Back to top](#table-of-contents)

---

## 26. Changing the query

> Deck: slide 20

Now change the query text to something else.

```python
query_text = "How is the economy doing?"
```

Run it again. The top two results come back as the stock market sentence and the investor sentence, because economy sits close to both. Both sit close to each other, and those are the top two scores. The others are not related.

You can see that whatever text you pass always gives you different results.

If you want to make it better, you can put the class inside a separate file, import that file here, and use it. You can obviously make it better than this. At the core, this is how it works.

[Back to top](#table-of-contents)

---

## 27. Making it survive a restart

> Deck: slide 23

The next step is storing this in a file.

If I run the file again, the vector store is gone and it creates the vector store again. If you do not want that, you can dump this into a JSON file, and load that JSON file back again.

```python
import json


def save_to_file(store, filepath):
    with open(filepath, "w") as f:
        json.dump(store.items, f)


def load_from_file(filepath):
    store = VectorStore()
    with open(filepath, "r") as f:
        store.items = json.load(f)
    return store
```

`json` ships with Python, so this costs you nothing.

[Back to top](#table-of-contents)

---

## 28. Where this goes next

> Deck: slides 21 and 22

You can take this to many more items. Five, two thousand, two million, depending on how you want to use it. It works in a similar way and it gives you all the scores. Obviously much more complex code has to go in, but that is how it works.

It searches for the nearest neighbour like this, and you make it more complex based on how fast you want it to be.

There are other vector stores available. FAISS is by Meta. Chroma DB is there. Qdrant is there, and Pinecone is there. Some of them are open source, so we can use them.

Whatever we are doing here, they are already doing through those libraries. All those functions sit inside them already, built in a very efficient manner, so we can use them directly in our own projects.

Later on, when you build something on a vector store, you are either going to use Chroma DB or FAISS. The difference is that now you know what they are replacing.

[Back to top](#table-of-contents)

---

## 29. One question from the session

> Deck: slide 06

**Does the vector length change with the input?**

That depends on the algorithm you use to convert into a vector representation, and it stays fixed. If we use a fixed size, the model understands it much better.

We use the same algorithm to convert the user query and the database into vector representation, so both follow a common embedding model. How many numbers you get depends on how complex the model is and how it was designed.

[Back to top](#table-of-contents)

---

## The full file

```python
import math
from sentence_transformers import SentenceTransformer


def dot_product(vector_a, vector_b):
    total = 0.0
    for a, b in zip(vector_a, vector_b):
        total += a * b
    return total


def magnitude(vector):
    total = 0.0
    for value in vector:
        total += value ** 2
    return math.sqrt(total)


def cosine_similarity(vector_a, vector_b):
    dot = dot_product(vector_a, vector_b)
    mag_a = magnitude(vector_a)
    mag_b = magnitude(vector_b)

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


class VectorStore:
    def __init__(self):
        self.items = []

    def add(self, text, vector):
        self.items.append({
            "text": text,
            "vector": list(vector),
        })

    def query(self, query_vector, top_k=2):
        scored_items = []

        for item in self.items:
            score = cosine_similarity(query_vector, item["vector"])
            scored_items.append((score, item["text"]))

        scored_items.sort(key=lambda pair: pair[0], reverse=True)
        return scored_items[:top_k]


model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat.",
    "The dog is playing in the park.",
    "The stock market crashed today.",
    "Investors are worried about inflation.",
    "The kitten is sleeping on the sofa.",
]

store = VectorStore()

for sentence in sentences:
    vector = model.encode(sentence)
    store.add(sentence, vector)

query_text = "A kitten is playing with a toy."
query_vector = model.encode(query_text)

results = store.query(query_vector, top_k=2)

print(f"Query: {query_text}")
print("Top 2 results:")
for score, text in results:
    print(f"  {score:.4f}  -  {text}")
```

[Back to top](#table-of-contents)

---

## Keep going

This session is a glimpse of what happens in **Build 30**. Build 30 is a challenge I started. It runs for 30 days, and every day we build and push our code to GitHub, so you prepare your portfolio along the way. You can join it as a live cohort, or one to one, and all the content from the 30 days sits inside both.

The core idea behind Buildership is depth. You learn how individual topics work, and then you build something with them, rather than keeping a brief about everything.

**AI Engineer HQ** runs its next cohort in October.

**Free resources.** The handbooks and eBooks are free to download, including the Tensor handbook, which explains tensors step by step.

Find all of it at [masterdexter.io](https://masterdexter.io), and the code for this session in the [AI Engineer Headquarters](https://github.com/hemansnation/AI-Engineer-Headquarters) repo.

---

*Free to read, free to share, free to fork.*