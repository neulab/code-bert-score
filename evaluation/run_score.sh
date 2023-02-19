# java 
python run_score.py \
    --lang java \
    --model neulab/codebert-java \
    --device cuda:0 \
    --d_folder data/humaneval_java_davinci-0.8-keep \
    --d_prefix humaneval \
    --idf_path data/idf/java_idf.pkl \
    --layer 7


# cpp
python run_score.py \
    --lang cpp \
    --model neulab/codebert-cpp \
    --device cuda:0 \
    --d_folder data/humaneval_cpp_davinci-0.8-keep \
    --d_prefix humaneval \
    --idf_path data/idf/cpp_idf.pkl \
    --layer 10

# js
python run_score.py \
    --lang cpp \
    --model neulab/codebert-javascript \
    --device cuda:0 \
    --d_folder data/humaneval_js_davinci-0.8-keep \
    --d_prefix humaneval \
    --idf_path data/idf/js_idf.pkl \
    --layer 10

# python 
# js
python run_score.py \
    --lang cpp \
    --model neulab/codebert-python \
    --device cuda:0 \
    --d_folder data/humaneval_python_davinci-0.8-keep \
    --d_prefix humaneval \
    --idf_path data/idf/python_idf.pkl \
    --layer 11