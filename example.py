import code_bert_score
import pickle
from nltk.translate import bleu_score
from nltk.translate.bleu_score import sentence_bleu
import re

# print('Sanity check:')
# precision, recall, f1, f3 = code_bert_score.score(cands=['void main() {'], refs=['void main() {'], lang='c')
# print(f'P: {precision} R: {recall} F1: {f1}, F3: {f3}')


# print('Examples for decompiled code:')

# refs = ['int posix_spawn_file_actions_init ( posix_spawn_file_actions_t * fa ) { fa -> __actions = 0 ; return 0 ; }',
#     'int strncmp ( const char * _l , const char * _r , size_t n ) { const unsigned char * l = ( void * ) _l ; const unsigned char * r = ( void * ) _r ; if ( ! ( n -- ) ) return 0 ; for ( ; ( ( ( * l ) && ( * r ) ) && n ) && ( ( * l ) == ( * r ) ) ; l ++ , r ++ , n -- ) ; return ( * l ) - ( * r ) ; }',
#     'CK_RV a_decrypt ( bee b , CK_BYTE_PTR cipher , CK_ULONG len , CK_OBJECT_HANDLE key , CK_BYTE_PTR * clear , CK_ULONG_PTR out_len ) { return decrypt_with_mechanism ( b , cipher , len , key , b . default_asym_m , clear , out_len ) ; }'
# ]

# predictions = [
#     'int posix_spawn_file_actions_init ( posix_t * t ) { t -> end = 0 ; return 0 ; }',
#     'int strncmp ( const char * s1 , const char * s2 , size_t n ) { if ( n == 0 ) return 0 ; while ( ( ( ( * s1 ) && ( * s2 ) ) && ( n > 0 ) ) && ( ( * s1 ) == ( * s2 ) ) ) { s1 ++ ; s2 ++ ; n -- ; } return ( ( unsigned char ) ( * s1 ) ) - ( ( unsigned char ) ( * s2 ) ) ; }',
#     'void a_decrypt ( KEY m1 , KEY k1 , KEY k2 ) { decrypt_with_mechanism ( m1 , k1 , k2 , k2 ) ; }'
# ]

# decompiled = [
#     'long long posix_spawn_file_actions_init ( long long a1 ) { * ( _QWORD * ) ( a1 + 8 ) = 0LL ; return 0LL ; }',
#     'long long strncmp ( _BYTE * a1 , _BYTE * a2 , long long a3 ) { signed long long v4 ; _BYTE * v5 ; _BYTE * v6 ; v6 = a1 ; v5 = a2 ; v4 = a3 - 1 ; if ( ! a3 ) return 0LL ; while ( * v6 && * v5 && v4 && * v6 == * v5 ) { ++ v6 ; ++ v5 ; -- v4 ; } return ( unsigned char ) * v6 - ( unsigned int ) ( unsigned char ) * v5 ; }',
#     'long long a_decrypt ( long long a1 , long long a2 , long long a3 , long long a4 , long long a5 , long long a6 , long long a7 , long long a8 , long long a9 , long long a10 , long long a11 , long long a12 ) { return decrypt_with_mechanism ( a1 , a2 , a3 , a12 ) ; }'
# ]

def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    tokens = [t for t in code.split(' ') if t]
    return tokens

def eval(predictions, decompiled, refs, **kwargs):
    # print('Predictions:')
    pred_results = code_bert_score.score(cands=predictions, refs=refs, lang='c', **kwargs)

    # print('Decompiled:')
    decompiled_results = code_bert_score.score(cands=decompiled, refs=refs, lang='c', **kwargs)

    for i in range(len(refs)):
        print(f'Example {i}:')
        print(f'Prediction precision: {pred_results[0][i]}, recall: {pred_results[1][i]}, f1: {pred_results[2][i]}, f3: {pred_results[3][i]}')
        print(f'Decompiled precision: {decompiled_results[0][i]}, recall: {decompiled_results[1][i]}, f1: {decompiled_results[2][i]}, f3: {decompiled_results[3][i]}')
        print()

# print('Default evaluation:')
# eval(predictions, decompiled, refs)

# print('Remove punctiation-only tokens after encoding:')
# eval(predictions, decompiled, refs, no_punc=True)

# print('Test long inputs (the model will chunk the inputs with overlap, and concatenate the outputs):')
# eval([' '.join(predictions) * 5], [' '.join(decompiled) * 5], [' '.join(refs) * 5])

# print('Test with sources:')
# eval(predictions, decompiled, refs, sources=['// Init fa actions', '// compare two strings', '// Decrypt'])

# print('Test with sources and no_punc:')
# eval(predictions, decompiled, refs, sources=['// Init fa actions', '// compare two strings', '// Decrypt'], no_punc=True)

with open('java_idf.pkl', 'rb') as f:
    java_idf = pickle.load(f)
# eval(predictions, decompiled, refs, no_punc=True, sources=['// Init fa actions', '// compare two strings', '// Decrypt'], idf=java_idf,)

predictions = ["""boolean f(Object target) {
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

pred_results = code_bert_score.score(cands=predictions, refs=refs, no_punc=True, lang='java', idf=java_idf)

for i in range(len(refs)):
    print(f'Example {i}:')
    print(f'Prediction precision: {pred_results[0][i]}, recall: {pred_results[1][i]}, f1: {pred_results[2][i]}, f3: {pred_results[3][i]}')
    print()
    ref_tokens = tokenize_for_bleu_eval(refs[i])
    pred_tokens = tokenize_for_bleu_eval(predictions[i])
    print(f'BLEU score: {sentence_bleu([ref_tokens], pred_tokens)}')