import code_bert_score
import pickle
from nltk.translate import bleu_score
from nltk.translate.bleu_score import sentence_bleu
import re

def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    tokens = [t for t in code.split(' ') if t]
    return tokens

def print_results(predictions, refs, pred_results):
    for i in range(len(refs)):
        print(f'Example {i}:')
        print(f'Reference: {refs[i]}')
        print(f'Prediction: {predictions[i]}')
        print(f'Prediction precision: {pred_results[0][i]:.3f}, recall: {pred_results[1][i]:.3f}, f1: {pred_results[2][i]:.3f}, f3: {pred_results[3][i]:.3f}')
        ref_tokens = tokenize_for_bleu_eval(refs[i])
        pred_tokens = tokenize_for_bleu_eval(predictions[i])
        print(f'BLEU score: {sentence_bleu([ref_tokens], pred_tokens):.3f}')
        print()

if __name__ == '__main__':
    predictions = [
"""boolean f(Object target) {
    for (Object elem: this.elements) {
        if (elem.equals(target)) {
            return true;
        }
    }
    return false;
}""", 
"""int f(Object target) {
    for (int i=0; i<this.elements.size(); i++) {
        Object elem = this.elements.get(i);
        if (elem.equals(target)) {
            return i;
        }
    }
    return -1;
}"""
    ]

    refs = [ \
"""int f(Object target) {
    int i = 0;
    for (Object elem: this.elements) {
        if (elem.equals(target)) {
            return i;
        }
        i++;
    }
    return -1;
}"""] * len(predictions)

    with open('idf_dicts/java_idf.pkl', 'rb') as f:
        java_idf = pickle.load(f)

    pred_results = code_bert_score.score([''],['a'], sources=["a"], lang="python")
    pred_results = code_bert_score.score(cands=predictions, refs=refs, no_punc=True, lang='java', idf=java_idf)
    print_results(predictions, refs, pred_results)

    print('When providing the context: "find the index of target in this.elements"')
    pred_results = code_bert_score.score(cands=predictions, refs=refs, no_punc=True, lang='java', idf=java_idf, sources=['find the index of target in this.elements'] * 2)
    print_results(predictions, refs, pred_results)