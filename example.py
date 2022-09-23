import code_bert_score

print('Sanity check:')
precision, recall, f1 = code_bert_score.score(cands=['void main() {'], refs=['void main() {'], lang='c')
print(f'P: {precision} R: {recall} F1: {f1}')


print('Examples from Luke:')

refs = ['int posix_spawn_file_actions_init ( posix_spawn_file_actions_t * fa ) { fa -> __actions = 0 ; return 0 ; }',
    'int strncmp ( const char * _l , const char * _r , size_t n ) { const unsigned char * l = ( void * ) _l ; const unsigned char * r = ( void * ) _r ; if ( ! ( n -- ) ) return 0 ; for ( ; ( ( ( * l ) && ( * r ) ) && n ) && ( ( * l ) == ( * r ) ) ; l ++ , r ++ , n -- ) ; return ( * l ) - ( * r ) ; }',
    'CK_RV a_decrypt ( bee b , CK_BYTE_PTR cipher , CK_ULONG len , CK_OBJECT_HANDLE key , CK_BYTE_PTR * clear , CK_ULONG_PTR out_len ) { return decrypt_with_mechanism ( b , cipher , len , key , b . default_asym_m , clear , out_len ) ; }'
]

predictions = [
    'int posix_spawn_file_actions_init ( posix_t * t ) { t -> end = 0 ; return 0 ; }',
    'int strncmp ( const char * s1 , const char * s2 , size_t n ) { if ( n == 0 ) return 0 ; while ( ( ( ( * s1 ) && ( * s2 ) ) && ( n > 0 ) ) && ( ( * s1 ) == ( * s2 ) ) ) { s1 ++ ; s2 ++ ; n -- ; } return ( ( unsigned char ) ( * s1 ) ) - ( ( unsigned char ) ( * s2 ) ) ; }',
    'void a_decrypt ( KEY m1 , KEY k1 , KEY k2 ) { decrypt_with_mechanism ( m1 , k1 , k2 , k2 ) ; }'
]

decompiled = [
    'long long posix_spawn_file_actions_init ( long long a1 ) { * ( _QWORD * ) ( a1 + 8 ) = 0LL ; return 0LL ; }',
    'long long strncmp ( _BYTE * a1 , _BYTE * a2 , long long a3 ) { signed long long v4 ; _BYTE * v5 ; _BYTE * v6 ; v6 = a1 ; v5 = a2 ; v4 = a3 - 1 ; if ( ! a3 ) return 0LL ; while ( * v6 && * v5 && v4 && * v6 == * v5 ) { ++ v6 ; ++ v5 ; -- v4 ; } return ( unsigned char ) * v6 - ( unsigned int ) ( unsigned char ) * v5 ; }',
    'long long a_decrypt ( long long a1 , long long a2 , long long a3 , long long a4 , long long a5 , long long a6 , long long a7 , long long a8 , long long a9 , long long a10 , long long a11 , long long a12 ) { return decrypt_with_mechanism ( a1 , a2 , a3 , a12 ) ; }'
]

def eval(predictions, decompiled, refs, **kwargs):
    # print('Predictions:')
    pred_results = code_bert_score.score(cands=predictions, refs=refs, lang='c', **kwargs)

    # print('Decompiled:')
    decompiled_results = code_bert_score.score(cands=decompiled, refs=refs, lang='c', **kwargs)

    for i in range(len(refs)):
        print(f'Example {i}:')
        print(f'Prediction precision: {pred_results[0][i]}, recall: {pred_results[1][i]}, f1: {pred_results[2][i]}')
        print(f'Decompiled precision: {decompiled_results[0][i]}, recall: {decompiled_results[1][i]}, f1: {decompiled_results[2][i]}')
        print()

print('Default evaluation:')
eval(predictions, decompiled, refs)

print('Remove punctiation-only tokens after encoding:')
eval(predictions, decompiled, refs, no_punc=True)

print('Test long inputs (the model will chunk the inputs with overlap, and concatenate the outputs):')
eval([' '.join(predictions) * 5], [' '.join(decompiled) * 5], [' '.join(refs) * 5])