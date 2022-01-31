<p align="center">
    <br>
    <img src="docs/source/images/simplify_logo.png" width="400"/>
    <br>
<p>

# Simplify

Everything you need for text simplification, reconstruct phrasal texture, rephrase and increase readability.




These are techniques for simplifiying sentences:

1. [Discourse](https://github.com/Lambda-3/DiscourseSimplification)
2. [OpenIE6](https://github.com/dair-iitd/openie6)
3. [BiSECT](https://github.com/mounicam/BiSECT) (to integrate)
4. [Lexical simplification](https://github.com/mounicam/lexical_simplification) (to integrate)
5. [Controllable Simplification](https://github.com/mounicam/controllable_simplification) (to integrate)
6. [Neural Text Simplification](https://github.com/senisioi/NeuralTextSimplification) (to integrate)



## Evaluators

Metrics to measure the performance of sentence simplifications:

1. Sari
2. Bleu

## Install

```shell
git clone https://github.com/sadakmed/simplify.git
cd simplify/
pip install -r requirements.txt
pip install -e .
```
## Use
    
```python
from simplify.simplifiers import Discourse
discoure = Discourse()
simple_output = discourse(complex_sentence_list)
```
    
# Citations
```
@inproceedings{kolluru&al20,
    title = "{O}pen{IE}6: {I}terative {G}rid {L}abeling and {C}oordination {A}nalysis for {O}pen {I}nformation {E}xtraction",\
    author = "Kolluru, Keshav  and
      Adlakha, Vaibhav and
      Aggarwal, Samarth and
      Mausam, and
      Chakrabarti, Soumen",
    booktitle = "The 58th Annual Meeting of the Association for Computational Linguistics (ACL)",
    month = July,
    year = "2020",
    address = {Seattle, U.S.A}
}


```
