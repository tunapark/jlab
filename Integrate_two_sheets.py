#Jlab 파일 오픈에 사용되는 라이브러리 
from utils import *
from konlpy.tag import Okt
import csv
import pandas as pd
import itertools
import numpy as np
from tqdm.notebook import tqdm
step1_output_name='시트합치기_step1.xlsx'
step2_output_name='시트합치기_step2.xlsx'

def adding_new_words_on_origin_sheet_step1 (oldfilename,newfilename):
    #1. 데이터 읽기 
    old_df=pd.read_excel(oldfilename)
    old_df=old_df.sort_values(by='Unnamed: 0').reset_index()
    old_df.drop('index',axis=1, inplace=True)
    new_df=pd.read_excel(newfilename)
    

    #2. 기준이 되는 단어 리스트 만들기 
    #new_df_T: 0행이 기준 단어가 됨!
    #old_df_T: standard_word행이 기준 단어가 됨!
    old_stand_word=[]
    old_stand_word=old_df['Unnamed: 0'].tolist()
    new_stand_word = new_df['0'].tolist()

    #3. 데이터프레임 전치시키기 
    old_df_T= old_df.T
    new_df_T= new_df.T
    
    tqdm.pandas()
    #4. 최종 합친 데이터프레임 만들기위해 옛날 시트부터 일단 복사
    df=old_df_T.copy()

    #5. old에서 new랑 같은 단어의 인덱스 (행번호) 찾아서 new sheet의 단어 중 old에 없는 것 추가
    for j, old_word in tqdm(enumerate(old_stand_word)):
        newlist= old_df_T[old_df_T[j].notna()][j].tolist() 
        for i, new_word in enumerate(new_stand_word):
            for m in newlist:
                if new_word == m:
                    #print('일치하는 단어 발견')
                    for k in new_df_T[new_df_T[i].notna()][i].tolist():
                        if k not in newlist:
                            #print(old_word,'에 단어추가:',k)
                            newlist.append(k)
                    df[j]=pd.Series(newlist, index=['Unnamed: '+str(x) for x in range(len(newlist))])

            #print(new_word, old_word)
            #print('일치하는 단어가 없습니다')

    #6. new+old 중복 없이 합친 파이널 데이터프레임 저장
    final_df= df.T
    final_df.to_excel( step1_output_name, index=False)

def adding_new_words_on_origin_sheet_step2(step1_output_name):
    #1. 뜻이 다른 단어 제거 
    df = pd.read_excel(step1_output_name, header=None )
    df_T= df.T
    step2_output_name = step1_output_name[:-11]+'_step2.xlsx'
    global step2_new_df
    step2_new_df=pd.DataFrame()
    def drop_unsimilar_words(row):
        global step2_new_df
        row = [x for x in row if pd.isnull(x) == False] #NaN없애기 위함 (type이 object인 NaN이더라.. )
        new_list=[] #새롭게 담을 리스트 
        stand_word= row[0] #기준단어
        new_list.append(stand_word)

        word_list= [ x for x in row if x.isdigit() ==False]
        if len(row)>=2:
            for word in row : 
                if word[0]!= stand_word[0]:    
                        continue
                else:
                    #print('기준단어',row[0],'뜻이 같은 단어가 나왔습니다.',word)
                    if word not in new_list:
                        new_list.append(word)
        else:
            pass
        step2_new_df=step2_new_df.append(pd.Series(new_list), ignore_index=True)
        return new_list
    new_list=df_T.apply(drop_unsimilar_words)
    step2_new_df.to_excel(step2_output_name,header=None)

def adding_new_words_on_origin_sheet_step3(step2_output_name, output_):
    df = pd.read_excel(step2_output_name)
    df = df.sort_values(by=['Unnamed: 0'] ,ascending=True)
    df = df.reset_index(drop=True)
    df = df.T
    df =df[1:]
    
    new_df = pd.DataFrame()
    try :
        #삭제할 인덱스 행 저장용 리스트 
        delete_index_list=[]
        delete_word_list=[]
        for index in (df):    
            #기준이 되는 행
            stand_row=[]
            stand_row = [x for x in df[index] if (pd.isnull(x) == False) and( str(x).isdigit() ==False)]  
            new_word_list=stand_row

            if index <2:
                new_df=new_df.append(pd.Series(new_word_list), ignore_index=True)
                continue
            elif index >len(df.T):
                break
            else:
                for i in [-2,-1,1,2]:
                    compare_row=[]
                    #비교할 행 (앞, 뒤 3 행씩)
                    compare_row= [x for x in df[index+i]if (pd.isnull(x) == False) and( str(x).isdigit() ==False)] #NaN없애기 위함 (type이 object인 NaN이더라.. )

                    for stand_word in stand_row:
                        for compare_word in compare_row:
                            #print('비포',new_word_list)
                            if stand_word==compare_word:
                                #print('같은 단어 있음.')

                                if index < index+i:
                                    front_index= index
                                    if (compare_word not in delete_word_list) and ((index+i) not in delete_index_list) :
                                        #print('중복 인덱스 추가',(index+i))
                                        delete_index_list.append(index+i)
                                        delete_word_list.append(compare_word)
                                    break
                                elif index > index+i:
                                    front_index= index+i
                                    if (compare_word not in delete_word_list) and ((index) not in delete_index_list) :
                                        #print('중복 인덱스 추가',(index))
                                        delete_index_list.append(index)
                                        delete_word_list.append(compare_word)
                                    break

                                if m not in new_word_list:
                                    new_word_list.append(m)
                                #print('-----------------')
                                #print('추가된',new_word_list)

                            else:

                                continue
                            break


                if index in delete_index_list :
                    #print(stand_word,compare_word)
                    #new_df=new_df.append(pd.Series(new_word_list), ignore_index=True)
                    continue
                else:
                    #print('-----------------')
                    #print('최종', new_word_list)
                    new_df=new_df.append(pd.Series(new_word_list), ignore_index=True)

    except KeyError as e:
        print('두개의 시트를 합치는 과정을 완료하였습니다.')
        #print(e) #다른 에러라면 확인
        pass

    #컬럼명 바꾸기 (이후에 이게 기존의 표제어사전이 될테니까 그 형식에 맞춰줌)
    new_df.columns=['Unnamed: '+str(x) for x in range(0,len(new_df.iloc[0]))]

    #최종 길이가 1인행 삭제 
    new_no_one_len_df= pd.DataFrame()
    for i in range(len(new_df)):
        x=new_df.iloc[i]
        new_no_one_len_list=[x for x in x if pd.isnull(x) == False]
        if len(new_no_one_len_list) <2 :
            continue
        else:
            new_no_one_len_df=new_no_one_len_df.append(pd.Series(new_no_one_len_list), ignore_index=True)
    #print(new_no_one_len_df.columns)
    #0열을 기준으로 가나다순 정렬
    #new_no_one_len_df = new_no_one_len_df.sort_values(by=[0] ,ascending=True)
    new_no_one_len_df.to_excel(output_+'_최종처리완료한_표제어사전.xlsx',index=False)

def Integrate_two_sheets(username,prname):
    #step 0. Jlab dictionary엑셀에서 인풋 읽기 -------
    if (username == None) & (prname == None):
        input_directory=''
    else:    
        input_directory = "/".join([username, prname])
    ref_, input_, output_ = Read_Arg_(username, prname, "Integrate_two_sheets")
    
    filename1=input_.split(',')[0]
    filename2=input_.split(',')[1]
    
    adding_new_words_on_origin_sheet_step1(filename1, filename2)
    adding_new_words_on_origin_sheet_step2( step1_output_name)
    adding_new_words_on_origin_sheet_step3( step2_output_name,output_)
    
if __name__=='__main__':
    Integrate_two_sheets(None,None)
