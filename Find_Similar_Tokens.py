#Jlab 파일 오픈에 사용되는 라이브러리 
from utils import *
#빈도분석에 사용되는 라이브러리
from konlpy.tag import Okt
from collections import Counter
import csv
import pandas as pd
import itertools
#단어유사도에 사용되는 라이브러리
from gensim.models.word2vec import Word2Vec
import numpy as np
import time

def similar_word_model_making(filename):
    from tqdm.notebook import tqdm
    tqdm.pandas()
    #1. input 파일 읽기
    if 'csv' in filename:
        ori_df= pd.read_csv(filename)
        ori_df = ori_df[['contents']] #'contents' 행만 읽기 
        temp_file_name= filename[:-4]
    elif 'xlsx' in filename:
        ori_df= pd.read_excel(filename)
        ori_df = ori_df[['contents']] #'contents' 행만 읽기 
        temp_file_name= filename[:-5]
    else:
        print('파일 형식이 잘못되었습니다. csv, xlsx 파일만 가능합니다.')
    
    #txt파일로 만들기 
    ori_df.to_csv(temp_file_name+'.txt', index=False, header=False)
    
    #2.만들어진 txt 파일 읽어들이기
    f= open(temp_file_name+'.txt','r',encoding='UTF-8')
    lines= f.readlines()
    f.close()
    kkma= Okt()
    #3.단어유사도 학습을 위한 형태소 분석 후 데이터셋 생성 (2차원 리스트)
    dataset= [] 
    print('학습을 위한 단어 데이터셋 만들기 진행중입니다.')
    for i in tqdm(range(len(lines))):
        dataset.append(kkma.morphs(lines[i]))
    dataset = [[y for y in x if not y.isdigit()] for x in dataset] #숫자제외
    dataset = [[y for y in x if not len(y)==1] for x in dataset] #2자이상

    #4. 워드투벡터 모델 학습 
    model = Word2Vec(sentences = dataset,  window = 5, min_count = 5, workers = 4, sg = 1)
    
    #5. 워드투벡터 모델 저장 
    model.save(temp_file_name+".model")
    print('단어 유사도 모델이 저장되었습니다.')
    
def morphs_analysis(filename):
    from tqdm.notebook import tqdm
    tqdm.pandas()
    # 파일은 같은 경로에 있어야 하며 위의 코드로 불용어 제거를 마친 파일을 그대로 넣어 사용합니다.
    df = pd.read_excel(filename)
    df = df.dropna(axis=0)
    df['contents'] = df['contents'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

    okt = Okt()
    def morphs(item):
        item_disassembled = okt.morphs(item) # okt의 형태소 분석기를 사용하여 형태소를 분리합니다.
        return item_disassembled

    df = df.dropna(axis=0) # 혹시 모를 결측치를 제거합니다
    print('인풋 데이터 빈도분석이 시작되었습니다.')
    df['contents'] = df['contents'].progress_apply(morphs) #이 부분이 계속 에러나서 tqdm 없이..
    allwords = list(itertools.chain.from_iterable(df['contents']))
    freq = Counter(allwords).most_common()
    freqdict = pd.DataFrame(dict(freq), index = ['freq'])
    freqdict = freqdict.transpose()	#행 열 전환
    freqdict.to_excel('{} 빈도.xlsx'.format(filename[:-5]))
    print('%s 빈도.xlsx 생성됨'%filename[:-5])   

def similar_word_dataframe(filename,ref_, output_):
    #step1. 만든 모델 불러오기 
    from gensim.models.word2vec import Word2Vec
    from tqdm.notebook import tqdm
    tqdm.pandas()
    
    if 'xlsx' in filename : 
        temp_file_name= filename[:-5]
    elif 'csv' in filename:
        temp_file_name= filename[:-4]
    else: 
        print('인풋 파일 형식이 잘못되었습니다.')
    
    #step1. 학습한 모델 불러오기 
    model = Word2Vec.load(temp_file_name+'.model')

    #step2. 사용자 인풋(option), 단어 최소빈도수 받아서 단어 유사도 데이터프레임 뽑아내기 
    word_option = int(ref_.split(',')[0])
    #word_option : 0:모든품사, 1:동사,형용사, 2:명사
    word_min_count = int(ref_.split(',')[1])
    #word_min_count : 최소등장빈도 설정
    sim_word_count = int(ref_.split(',')[2])
    #sim_word_count : 유사한단어 몇개까지 뽑을것인지 설정
    
    #step3. 빈도 분석 파일 불러와서 상위 빈도 x개 (user input) 단어 리스트 만들기
    okt_pos = Okt()   # 형태소 분석 ( norm : 정규화 )
    index=0
    df= pd.DataFrame(columns=[str(x) for x in range(0,sim_word_count+1)])
    top_word_df= pd.read_excel(temp_file_name+' 빈도.xlsx')
    sliced_top_word_df = top_word_df[top_word_df['freq']>=word_min_count]
    top_word_list_temp= sliced_top_word_df['Unnamed: 0'].tolist()
    
    #step3-2. top_word_list에서 형태소 분석 진행 후, 같은 형태소 있으면 위에 걸로 하고, 아래거 없애기
    top_word_list=[] #중복없는 기준단어 넣을 리스트

    top_word_list_for_nondu=[]
    for word in (top_word_list_temp) : 
            if (okt_pos.pos(word, stem=True)[0][0]) not in top_word_list_for_nondu:
                #print(okt_pos.pos(word, stem=True)[0][0])
                top_word_list_for_nondu.append(okt_pos.pos(word, stem=True)[0][0])
                top_word_list.append(word)

    
    #step4. user input 받은 옵션(모든 품사:0 / 명사:1 / 동사,형용사:2)에 따라 유사한 단어 데이터프레임 생성
    print('유사한 단어 추출 진행중입니다.')
    for i in tqdm(top_word_list): 
        sim_word_list=[]
        try:
            model_result = model.wv.similar_by_vector(i, topn=sim_word_count, restrict_vocab=None)
            sim_word_list.append(i)    
            
            for j in range(len(model_result)):

                okt_w1=okt_pos.pos(i)[0][1]
                okt_w2=okt_pos.pos(model_result[j][0])[0][1]
                
                if word_option==1: 
                    output_name='동사, 형용사만 '+output_
                    if (okt_w1==okt_w2=='Verb') or  (okt_w1==okt_w2=='Adjective'):
                        if float(model_result[j][1])>=0.6: #유사도가 0.6이상인 것만 뽑음
                            sim_word_list.append(model_result[j][0])    
                    else:
                        continue
                elif word_option==2: 
                    output_name='명사만 '+output_
                    if (okt_w1==okt_w2=='Noun'):
                        if float(model_result[j][1])>=0.6: #유사도가 0.6이상인 것만 뽑음
                            sim_word_list.append(model_result[j][0])     
                    else:
                        continue
                else:
                    output_name='모든품사 '+output_
                    if (okt_w1==okt_w2):
                        if float(model_result[j][1])>=0.6: #유사도가 0.6이상인 것만 뽑음
                            sim_word_list.append(model_result[j][0])       
                    else:
                        continue
        except KeyError as e:
            #print(e)
            #print('-----이 단어는 학습할 당시 없었던 단어이기에 유사한 단어를 불러올 수 없어 스킵합니다.-----')
            continue
        if len(sim_word_list)<2:
            continue
        else:
            df = df.append(pd.Series(sim_word_list,index=df.columns[:len(sim_word_list)]), ignore_index=True)
    #오류나면 컬럼명 확인해보기 : print(df.columns())
    df = df[df['1'].notna()]
    #print(df.columns)
    df.to_excel(output_name, index=False, header=False)

def Find_Similar_Tokens(username,prname):
     #step 0. Jlab dictionary엑셀에서 인풋 읽기 -------
    if (username == None) & (prname == None):
        input_directory=''
    else:    
        input_directory = "/".join([username, prname])
    ref_, input_, output_ = Read_Arg_(username, prname, "Find_Similar_Tokens")
    
    filename=input_
    similar_word_model_making(filename)
    morphs_analysis(filename)
    similar_word_dataframe(filename, ref_,output_)
if __name__=='__main__': 
    
    Find_Similar_Tokens(None, None)
