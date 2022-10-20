import pandas as pd
from utils import * 
from Local_ver import *
import os, re
import pandas as pd
from tqdm import tqdm
from flashtext import KeywordProcessor
##파일 이름 바꾸기 



def Lemma_max_count_word(username, prname):
    #0. 파일 읽기 
    if (username == None) & (prname == None):
        input_directory=''
    else:    
        input_directory = "/".join([username, prname])
    ref, input_, output_ = Read_Arg_(username, prname, "Lemma_max_count_word")
    #1. 표제화를 하지 않은 리뷰 데이터 열어서 빈도분석을 돌린다 
    import pandas as pd 
    if input_[-3:]=='csv':
        df = pd.read_csv(input_)
    elif input_[-4:]=='xlsx':
        df = pd.read_excel(input_)
    else:
        print('잘못된 인풋파일 형식입니다.')
    count_df =Frequency_Analysis(input_, None)
    #count_df.to_excel('(불용어)스타벅스 빈도.xlsx')
    
    #2. 표제화 사전 열기 
    Lemmadict = Read_Sheet_(username,prname,ref)
    
    #3. 표제화 사전 전치하기
    Lemmadict_trans_df =Lemmadict.transpose()

    #4. 각각의 행에 대해서 표제화할 단어를 고르는 함수  
    def find_max(df_row):
        df_list = df_row.to_list()
        df_list = list(filter(None, df_list))
        word_dict={}
        for i in df_list: 
            try :
                word_dict[i]=count_df.loc[count_df['tag']==i,'count'].values[0]
            except :
                word_dict[i]=0
        max_key = max(word_dict, key=word_dict.get)
        print(word_dict) #확인용 출력문
        return max_key
    
    #5. max_word라는 새로운 열을 1열에 만들고 제일 빈도수가 높은 단어를 적는다  
    try :
        Lemmadict.insert(0,'max_word',Lemmadict_trans_df.apply(find_max)) 
    except ValueError:
        pass
    
    #6. 새로운 표제화 사전으로 빈도 표제화 수행
    def Replace_Texts_in_Messages():  
        #lemma : "JDic_Lemmatization(일반lemma사전)"시트를 불러옵니다.
        #input_Message : 표제화할 input 데이터 프레임 
        #input_directory :로컬이면 None
        #output_ : 표제화해서 내보낼 파일 이름 
        lemma = Lemmadict
        lemma = lemma.fillna("").to_numpy(dtype=list)
        all_V = list(map(lambda x: [i for i in x if i != ""], lemma))  # all_V라는 변수에 lemma에 있는 데이터들을 전부 가져옵니다.

        """
        version 2.0 token decomposition 방식(21-05-30)
        """
        # lemee에는 lemmatize될 token을, lemer에서는 기준 token을 추가해준다.
        # lemee와 lemerfmf 열로 갖는 DataFrame을 lemm이라는 변수에 담아둔다.
        lemee = []
        lemer = []
        for case in all_V:
            standardised = case[0]
            for keyword in case[1:]:
                lemee.append(keyword)
                lemer.append(standardised)
        lemm = pd.DataFrame({"raw": lemee, "lem": lemer})


        input_Message = import_dataframe(input_)

        # 원문 데이터로부 line넘버, token넘버고, token을 추출해
        # line_no, token_no, token 을 열로 갖는 DataFrame을 text_decomposition이라는 의미의 text_decomp 변수에 저장한다.
        line_no = []
        token_no = []
        token = []
        for lines in enumerate(input_Message["contents"]):
            for tokens in enumerate(str(lines[1]).split()):
                line_no.append(lines[0])
                token_no.append(tokens[0])
                token.append(tokens[1])
        text_decomp = pd.DataFrame({"line_no": line_no, "token_no": token_no, "token": token})

        # text_decomp 테이블 기준테이블로 설정하고  text_decomp의 "token"열과  lemm테이블의 "raw" 열을 "left join" 하고,
        #  중복열인 "raw"열을 제거한 후, "lem"열에 빈 부분을 같은 행의 "token"열의 값들로 채워준다.
        res = pd.merge(text_decomp, lemm, left_on=["token"], right_on=["raw"], how="left").drop(["raw"], axis=1)
        res["lem"] = res["lem"].fillna(res["token"])
        # 중간에 res가 어떻게 나오는지, 바뀐 부분이 어떻게 바뀌었는지 확인하는 코드 두 줄
        print(res.head(30))
        print(res[res["token"]!=res["lem"]].head(30))

        # lemmatize된 문장으로 뭉쳐주는 코드
        # new_lines라는 빈 리스트를 생성한다.

        # res의 line_no열에 있는 값들을 unique하게 불러온 후, 이들을 기준으로 순서대로 다음과 같은 실행을 거친다.
        #     res의 "line_no"가  i번째인 부분을 가져온 후, "token_no"를 기준으로 오름차순으로 정렬한 후 그 순서대로 "lem"열에 있는 token들을 정렬한다.
        #     정렬된 token 사이를 띄어쓰기로 채워 넣어 한 문장으로 만들어 new_line이라는 변수에 저장한다.
        #     new_lines리스트에 new_line을 추가한다.
        res = res.sort_values(by=["line_no","token_no"])
        renewed = res.groupby("line_no", as_index=False).agg({'lem': ' '.join})["lem"]
        input_Message["contents"] = renewed


        # sen_no = res["line_no"].unique()
        #
        # for i in tqdm(sen_no):
        #     new_line = " ".join(res[res["line_no"] == i].sort_values(by="token_no", ascending=True)["lem"])
        #     new_lines.append(new_line)qs
        # # 최종적으로 누적된 new_lines를 input_Message의 "contents"열에 넣어 갱신해준다.
        # input_Message["contents"] = new_lines

        output_name = os.path.join(input_directory, output_)
        export_dataframe(input_Message, output_name)

        return input_Message  # 처리한 input_Message를 리턴값으로 내보냅니다.

    #7. 표제화한 데이터저장, 새로운 표제화 사전 저장 
    Replace_Texts_in_Messages().to_excel(output_)
    #Lemmadict.to_excel(output_)
if __name__=='__main__':
    Lemma_max_count_word(None, None)
    
